// backend/src/server.ts
import express, { Request, Response, NextFunction } from "express";
import bodyParser from "body-parser";
import cors from "cors";
import fs from "fs";
import path from "path";
import pdfParse from "pdf-parse";
import mammoth from "mammoth";
import winston from "winston";
import dotenv from "dotenv";

import { Ollama } from "@langchain/community/llms/ollama";
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { Document } from "langchain/document";
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import { PromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import {
  RunnableSequence,
  RunnablePassthrough,
} from "@langchain/core/runnables";

dotenv.config();

// --- Config
const PORT = Number(process.env.PORT) || 5000;
const PUBLIC_FOLDER = process.env.PUBLIC_FOLDER || "./public";
const EMBEDDING_MODEL = process.env.EMBEDDING_MODEL || "nomic-embed-text";
const LLM_MODEL = process.env.LLM_MODEL || "llama3.1";
const VECTOR_STORE_PATH = "./vector_store";

// --- Logger setup
const logger = winston.createLogger({
  level: "info",
  format: winston.format.printf(({ level, message }) => {
    const colors: Record<string, string> = {
      info: "\x1b[34m",
      warn: "\x1b[33m",
      error: "\x1b[31m",
    };
    const color = colors[level] || "\x1b[0m";
    return `${color}[${level.toUpperCase()}]\x1b[0m: ${message}`;
  }),
  transports: [
    new winston.transports.Console(),
    new winston.transports.File({ filename: "logs/server.log" }),
  ],
});

// --- Express setup
const app = express();
app.use(cors());
app.use(bodyParser.json());
app.use((req: Request, _res: Response, next: NextFunction) => {
  const methodColors: Record<string, string> = {
    GET: "\x1b[34m",
    POST: "\x1b[32m",
    PUT: "\x1b[33m",
    DELETE: "\x1b[31m",
  };
  const color = methodColors[req.method] || "\x1b[0m";
  console.log(
    `[${color}${req.method}\x1b[0m]: ${req.url} ${
      req.method === "POST" ? JSON.stringify(req.body) : ""
    }`
  );
  next();
});

// --- Vector store
let vectorStore: HNSWLib | null = null;

// --- File helpers
const fileExists = (p: string) => fs.existsSync(p);
const isSupportedFile = (filePath: string) =>
  [".pdf", ".txt", ".docx"].includes(path.extname(filePath).toLowerCase());
const getAllFiles = (folderPath: string): string[] =>
  fileExists(folderPath)
    ? fs.readdirSync(folderPath, { withFileTypes: true }).flatMap((item) => {
        const fullPath = path.join(folderPath, item.name);
        return item.isDirectory()
          ? getAllFiles(fullPath)
          : item.isFile() && isSupportedFile(fullPath)
          ? [fullPath]
          : [];
      })
    : [];

// --- Process file to text chunks
async function processFile(filePath: string): Promise<Document[]> {
  try {
    const fileName = path.basename(filePath);
    console.log(`\x1b[32mProcessing file:\x1b[0m ${fileName}`); // Green text

    let text = "";
    const ext = path.extname(filePath).toLowerCase();

    if (ext === ".pdf") text = (await pdfParse(fs.readFileSync(filePath))).text;
    else if (ext === ".txt") text = fs.readFileSync(filePath, "utf-8");
    else if (ext === ".docx")
      text = (await mammoth.extractRawText({ path: filePath })).value;

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 500,
      chunkOverlap: 50,
    });

    // Await the Promise before iterating
    const chunks = await splitter.splitDocuments([
      new Document({
        pageContent: text,
        metadata: { source: fileName },
      }),
    ]);

    // Log each chunk's source file
    chunks.forEach((chunk, idx) =>
      console.log(`\x1b[36m[Chunk ${idx + 1}]\x1b[0m from ${fileName}`)
    );

    return chunks;
  } catch (err) {
    logger.error(`Failed to process ${filePath}: ${(err as Error).message}`);
    return [];
  }
}

// --- Initialize knowledge base
async function initializeKnowledgeBase() {
  const embeddings = new OllamaEmbeddings({ model: EMBEDDING_MODEL });
  if (!fileExists(PUBLIC_FOLDER))
    fs.mkdirSync(PUBLIC_FOLDER, { recursive: true });

  if (fileExists(VECTOR_STORE_PATH)) {
    logger.info("Loading existing vector store...");
    vectorStore = await HNSWLib.load(VECTOR_STORE_PATH, embeddings);
    logger.info("Vector store loaded.");
    return;
  }

  const files = getAllFiles(PUBLIC_FOLDER);
  if (!files.length) {
    logger.warn("No files found in PUBLIC_FOLDER.");
    return;
  }

  logger.info(`Found ${files.length} files. Processing...`);
  const allDocs = (await Promise.all(files.map(processFile))).flat();
  vectorStore = await HNSWLib.fromDocuments(allDocs, embeddings);
  await vectorStore.save(VECTOR_STORE_PATH);
  logger.info(`Knowledge base initialized with ${allDocs.length} chunks.`);
}

// --- Chat handler
async function handleChat(message: string): Promise<string> {
  const llm = new Ollama({ model: LLM_MODEL, temperature: 0.7 });
  const userWantsDocs = /fetch|search|documents?|pdf|txt|docx/i.test(message);

  if (vectorStore && userWantsDocs) {
    const retriever = vectorStore.asRetriever();
    const relevantDocs = await retriever.getRelevantDocuments(message);
    if (relevantDocs.length) {
      const contextText = relevantDocs.map((d) => d.pageContent).join("\n\n");
      const prompt = PromptTemplate.fromTemplate(
        `Answer based on documents below. If answer not in docs, say so.\nContext:\n{context}\nQuestion:\n{question}\nAnswer:`
      );
      const chain = RunnableSequence.from([
        {
          context: new RunnablePassthrough(),
          question: new RunnablePassthrough(),
        },
        prompt,
        llm,
        new StringOutputParser(),
      ]);
      return chain.invoke({ context: contextText, question: message });
    } else {
      return "I could not find relevant documents. Ask me anything else!";
    }
  }

  // Normal chatbot response
  return llm.call(
    `You are a helpful AI assistant.\nUser: ${message}\nAssistant:`
  );
}

// --- Express endpoint
app.post(
  "/chat",
  asyncHandler(async (req: Request, res: Response) => {
    const { message } = req.body;
    if (!message)
      return res.status(400).json({ error: "Message is required." });
    const reply = await handleChat(message);
    res.json({ reply });
  })
);

// --- Async handler wrapper
function asyncHandler(fn: Function) {
  return (req: Request, res: Response, next: NextFunction) =>
    Promise.resolve(fn(req, res, next)).catch(next);
}

// --- Global error handler
app.use((err: Error, _req: Request, res: Response, _next: NextFunction) => {
  console.error(`\x1b[31m[ERROR]\x1b[0m: ${err.message}`);
  res.status(500).json({ error: err.message });
});

// --- Start server
initializeKnowledgeBase()
  .then(() =>
    app.listen(PORT, () =>
      logger.info(`Server running at http://localhost:${PORT}`)
    )
  )
  .catch((err) => {
    logger.error(`Fatal startup error: ${(err as Error).message}`);
    process.exit(1);
  });
