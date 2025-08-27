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
import crypto from "crypto";

import { Ollama } from "@langchain/community/llms/ollama";
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { Document } from "langchain/document";
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import { PromptTemplate } from "@langchain/core/prompts";
import { tools } from "./tools";
import pkg from "../package.json";

dotenv.config();

// --- Config
const PORT = Number(process.env.PORT) || 5000;
const PUBLIC_FOLDER = process.env.PUBLIC_FOLDER || "./public";
const VECTOR_STORE_PATH = "./vector_store";
const EMBEDDING_MODEL = process.env.EMBEDDING_MODEL || "nomic-embed-text";
const LLM_MODEL = process.env.LLM_MODEL || "llama3.1";
const BATCH_SIZE = 5;
const TOP_K = 5;
const SUPPORTED_FILE_FORMATS = (
  process.env.SUPPORTED_FILE_FORMATS || ".pdf,.txt,.docx,.csv,.html,.md"
)
  .split(",")
  .map((f) => f.trim().toLowerCase());

// --- Logger
const logger = winston.createLogger({
  level: "info",
  format: winston.format.printf(({ level, message }) => {
    const colors: Record<string, string> = {
      info: "\x1b[34m",
      warn: "\x1b[33m",
      error: "\x1b[31m",
    };
    return `${colors[level] || ""}[${level.toUpperCase()}]\x1b[0m ${message}`;
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

// --- Vector store (lazy init)
let vectorStore: HNSWLib | null = null;

// --- Helpers
const fileExists = (p: string) => fs.existsSync(p);
const isSupportedFile = (filePath: string) =>
  SUPPORTED_FILE_FORMATS.includes(path.extname(filePath).toLowerCase());

const getAllFiles = (folderPath: string): string[] => {
  if (!fileExists(folderPath)) return [];
  return fs.readdirSync(folderPath, { withFileTypes: true }).flatMap((item) => {
    const fullPath = path.join(folderPath, item.name);
    return item.isDirectory()
      ? getAllFiles(fullPath)
      : item.isFile() && isSupportedFile(fullPath)
      ? [fullPath]
      : [];
  });
};

// --- Process file into docs
async function processFile(filePath: string): Promise<Document[]> {
  const fileName = path.basename(filePath);
  let text = "";

  try {
    const ext = path.extname(filePath).toLowerCase();
    const fileBuffer = fs.readFileSync(filePath);

    switch (ext) {
      case ".pdf":
        text = (await pdfParse(fileBuffer)).text;
        break;
      case ".txt":
      case ".md":
        text = fileBuffer.toString("utf-8");
        break;
      case ".docx":
        text = (await mammoth.extractRawText({ path: filePath })).value;
        break;
      case ".csv":
        text = await new Promise<string>((resolve) => {
          const chunks: string[] = [];
          fs.createReadStream(filePath)
            .pipe(csvParser())
            .on("data", (row) => chunks.push(Object.values(row).join(" ")))
            .on("end", () => resolve(chunks.join("\n")));
        });
        break;
      case ".html":
        text =
          new JSDOM(fileBuffer.toString("utf-8")).window.document.body
            .textContent || "";
        break;
    }

    if (!text.trim()) return [];

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 500,
      chunkOverlap: 50,
    });

    return splitter.splitDocuments([
      new Document({ pageContent: text, metadata: { source: fileName } }),
    ]);
  } catch (err) {
    logger.error(`Failed to process ${filePath}: ${(err as Error).message}`);
    return [];
  }
}

// --- Initialize knowledge base
async function initializeKnowledgeBase() {
  try {
    const embeddings = new OllamaEmbeddings({ model: EMBEDDING_MODEL });

    if (fileExists(VECTOR_STORE_PATH)) {
      vectorStore = await HNSWLib.load(VECTOR_STORE_PATH, embeddings);
      logger.info("Vector store loaded from disk.");
      return;
    }

    const files = getAllFiles(PUBLIC_FOLDER);
    if (!files.length) {
      logger.warn("No files found in knowledge base folder.");
      return;
    }

    const allDocs = (
      await Promise.all(files.map((f) => processFile(f)))
    ).flat();

    if (!allDocs.length) {
      logger.warn("No documents processed for knowledge base.");
      return;
    }

    vectorStore = await HNSWLib.fromDocuments(allDocs, embeddings);
    await vectorStore.save(VECTOR_STORE_PATH);
    logger.info(`Knowledge base initialized with ${allDocs.length} chunks.`);
  } catch (err) {
    logger.error(`Knowledge base init failed: ${(err as Error).message}`);
  }
}

// --- Add chat replies to RAG
async function addToVectorStore(text: string, sourceName = "chat") {
  if (!vectorStore || !text.trim()) return;

  const hash = crypto.createHash("sha256").update(text).digest("hex");
  const docs = (vectorStore as any)?.docs || [];

  if (docs.some((d: Document) => d.metadata.hash === hash)) return;

  await vectorStore.addDocuments([
    new Document({ pageContent: text, metadata: { source: sourceName, hash } }),
  ]);
  await vectorStore.save(VECTOR_STORE_PATH);
}

// --- Chat handler
async function handleChat(message: string, useRAG = false) {
  const llm = new Ollama({ model: LLM_MODEL, temperature: 0.7 });

  let contextText = "";
  if (vectorStore && useRAG) {
    const retriever = vectorStore.asRetriever({ k: TOP_K });
    const docs = await retriever.invoke(message);
    contextText = docs.map((d) => d.pageContent.slice(0, 300)).join("\n\n");
  }

  const routerPrompt = PromptTemplate.fromTemplate(`
    You are a tool router.
    Decide which tool to use based on the question.

    ## Available tools:
    - codingTool → for programming problems.
    - default → for general reasoning.

    Question: {question}
    Respond ONLY with tool name.
  `);

  const toolChoice = (
    await llm.invoke(await routerPrompt.format({ question: message }))
  )
    ?.trim()
    .toLowerCase();

  const tool = tools[toolChoice] || tools.default;
  const reply = await tool.run({ question: message, context: contextText });

  await addToVectorStore(reply);

  return { reply, source: toolChoice };
}

// --- Routes
app.get("/vector-stats", (_req, res) => {
  res.json({
    totalVectors: (vectorStore as any)?.docs?.length || 0,
    files: getAllFiles(PUBLIC_FOLDER).length,
    topK: TOP_K,
    supportedFormats: SUPPORTED_FILE_FORMATS,
  });
});

app.post(
  "/chat",
  asyncHandler(async (req: Request, res: Response) => {
    const { message, useRAG } = req.body;
    if (!message)
      return res.status(400).json({ error: "Message is required." });

    const { reply, source } = await handleChat(message, useRAG);
    res.json({ reply, source });
  })
);

// --- Async wrapper
function asyncHandler(
  fn: (req: Request, res: Response, next: NextFunction) => Promise<any>
) {
  return (req: Request, res: Response, next: NextFunction) =>
    Promise.resolve(fn(req, res, next)).catch(next);
}

// --- Global error handler
app.use((err: Error, _req: Request, res: Response, _next: NextFunction) => {
  logger.error(err.message);
  res.status(500).json({ error: err.message });
});

// --- Start server
console.log(`\n${pkg.name} v${pkg.version}`);

initializeKnowledgeBase().finally(() => {
  app.listen(PORT, () => {
    logger.info("✅ Server started successfully");
    logger.info(`🌐 Running at: http://localhost:${PORT}`);
    logger.info(
      `⚙️ Node: ${process.version}, Embeddings: ${EMBEDDING_MODEL}, LLM: ${LLM_MODEL}`
    );
  });
});
