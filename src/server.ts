// backend/src/server.ts
import express, { Request, Response, NextFunction } from "express";
import bodyParser from "body-parser";
import cors from "cors";
import fs from "fs";
import path from "path";
import pdfParse from "pdf-parse";
import mammoth from "mammoth";
import csvParser from "csv-parser";
import { JSDOM } from "jsdom";
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
const BATCH_SIZE = 5; // batch processing chunks
const TOP_K = 5; // top retrieved documents
const SUPPORTED_FILE_FORMATS = (
  process.env.SUPPORTED_FILE_FORMATS || ".pdf,.txt,.docx,.csv,.html,.md"
)
  .split(",")
  .map((f) => f.trim().toLowerCase());

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
  SUPPORTED_FILE_FORMATS.includes(path.extname(filePath).toLowerCase());

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
    logger.info(`Processing file: ${fileName}`);

    let text = "";
    const ext = path.extname(filePath).toLowerCase();

    if (ext === ".pdf") text = (await pdfParse(fs.readFileSync(filePath))).text;
    else if (ext === ".txt") text = fs.readFileSync(filePath, "utf-8");
    else if (ext === ".docx")
      text = (await mammoth.extractRawText({ path: filePath })).value;
    else if (ext === ".csv") {
      text = "";
      await new Promise<void>((resolve) => {
        fs.createReadStream(filePath)
          .pipe(csvParser())
          .on("data", (row) => {
            text += Object.values(row).join(" ") + "\n";
          })
          .on("end", () => resolve());
      });
    } else if (ext === ".html") {
      const html = fs.readFileSync(filePath, "utf-8");
      text = new JSDOM(html).window.document.body.textContent || "";
    } else if (ext === ".md") text = fs.readFileSync(filePath, "utf-8");

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 500,
      chunkOverlap: 50,
    });

    const chunks = await splitter.splitDocuments([
      new Document({ pageContent: text, metadata: { source: fileName } }),
    ]);

    chunks.forEach((chunk, idx) =>
      logger.info(`[Chunk ${idx + 1}] from ${fileName}`)
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

  logger.info(`Found ${files.length} files. Processing in batches...`);

  let allDocs: Document[] = [];
  for (let i = 0; i < files.length; i += BATCH_SIZE) {
    const batchFiles = files.slice(i, i + BATCH_SIZE);
    const batchDocs = (await Promise.all(batchFiles.map(processFile))).flat();
    allDocs = allDocs.concat(batchDocs);
    logger.info(
      `Processed batch ${i / BATCH_SIZE + 1}, total chunks: ${allDocs.length}`
    );
  }

  vectorStore = await HNSWLib.fromDocuments(allDocs, embeddings);
  await vectorStore.save(VECTOR_STORE_PATH);
  logger.info(`Knowledge base initialized with ${allDocs.length} chunks.`);
}

// --- Chat handler
async function handleChat(
  message: string
): Promise<{ reply: string; source: "RAG" | "LLM" }> {
  const llm = new Ollama({
    model: LLM_MODEL,
    temperature: 0.7,
    callbacks: [
      {
        handleLLMNewToken(token: string) {
          logger.info(`[LLM TOKEN]: ${token}`);
        },
      },
    ],
  });

  if (vectorStore) {
    const retriever = vectorStore.asRetriever({ k: TOP_K });
    const relevantDocs = await retriever.getRelevantDocuments(message);

    if (relevantDocs.length) {
      const contextText = relevantDocs
        .map((d) => d.metadata.summary || d.pageContent.slice(0, 500))
        .join("\n\n");

      const prompt = PromptTemplate.fromTemplate(`
        You are an AI assistant. Use the provided context to answer the user's question. 
        If the answer cannot be found in the context, respond honestly that the information is not available.
        Context:
        {context}

        Question:
        {question}

        Answer:
      `);

      const chain = RunnableSequence.from([
        {
          context: new RunnablePassthrough(),
          question: new RunnablePassthrough(),
        },
        prompt,
        llm,
        new StringOutputParser(),
      ]);

      const reply = await chain.invoke({
        context: contextText,
        question: message,
      });
      return { reply, source: "RAG" }; // indicate RAG
    } else {
      return {
        reply: "I could not find relevant documents. Ask me anything else!",
        source: "RAG",
      };
    }
  }

  const reply = await llm.call(
    `You are a helpful AI assistant.\nUser: ${message}\nAssistant:`
  );
  return { reply, source: "LLM" }; // indicate direct LLM
}

// --- Vector store stats endpoint
app.get("/vector-stats", async (_req, res) => {
  if (!vectorStore)
    return res.status(404).json({ error: "Vector store not initialized." });
  const stats = {
    totalVectors: (vectorStore as any).docs?.length || 0,
    files: getAllFiles(PUBLIC_FOLDER).length,
    topK: TOP_K,
    supportedFormats: SUPPORTED_FILE_FORMATS,
  };
  res.json(stats);
});

// --- Express endpoint
app.post(
  "/chat",
  asyncHandler(async (req: Request, res: Response) => {
    const { message } = req.body;
    if (!message)
      return res.status(400).json({ error: "Message is required." });
    const { reply, source } = await handleChat(message);
    res.json({ reply, source });
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
