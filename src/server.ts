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

// --- Process file into docs
async function processFile(filePath: string): Promise<Document[]> {
  const fileName = path.basename(filePath);
  let text = "";

  try {
    switch (path.extname(filePath).toLowerCase()) {
      case ".pdf":
        text = (await pdfParse(fs.readFileSync(filePath))).text;
        break;
      case ".txt":
      case ".md":
        text = fs.readFileSync(filePath, "utf-8");
        break;
      case ".docx":
        text = (await mammoth.extractRawText({ path: filePath })).value;
        break;
      case ".csv":
        await new Promise<void>((resolve) => {
          const chunks: string[] = [];
          fs.createReadStream(filePath)
            .pipe(csvParser())
            .on("data", (row) => chunks.push(Object.values(row).join(" ")))
            .on("end", () => {
              text = chunks.join("\n");
              resolve();
            });
        });
        break;
      case ".html":
        text =
          new JSDOM(fs.readFileSync(filePath, "utf-8")).window.document.body
            .textContent || "";
        break;
    }

    if (!text.trim()) return [];

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 500,
      chunkOverlap: 50,
    });

    return await splitter.splitDocuments([
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
      logger.info("Vector store loaded.");
      return;
    }

    const files = getAllFiles(PUBLIC_FOLDER);
    if (!files.length) {
      logger.warn("No files found in knowledge base.");
      return;
    }

    const allDocs: Document[] = [];
    for (let i = 0; i < files.length; i += BATCH_SIZE) {
      const batchDocs = (
        await Promise.all(files.slice(i, i + BATCH_SIZE).map(processFile))
      ).flat();
      allDocs.push(...batchDocs);
    }

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
  if (
    (vectorStore as any).docs?.some((d: Document) => d.metadata.hash === hash)
  )
    return;

  await vectorStore.addDocuments([
    new Document({ pageContent: text, metadata: { source: sourceName, hash } }),
  ]);
  await vectorStore.save(VECTOR_STORE_PATH);
}

// --- Chat handler
async function handleChat(message: string, useRAG = false) {
  const llm = new Ollama({
    model: LLM_MODEL,
    temperature: 0.7,
    callbacks: [{ handleLLMNewToken: (t) => process.stdout.write(t) }],
  });

  let contextText = "";
  let source: "RAG" | "LLM" = "LLM";

  if (vectorStore && useRAG) {
    const retriever = vectorStore.asRetriever({ k: TOP_K });
    const docs = await retriever.invoke(message);
    if (docs.length) {
      contextText = docs.map((d) => d.pageContent.slice(0, 300)).join("\n\n");
      source = "RAG";
    }
  }

  const prompt = PromptTemplate.fromTemplate(`
    You are an advanced AI agent with access to multiple tools.
    All tools are implemented as separate modules inside the **tools/** folder.

    Available tools:
    - codingTool: For solving programming problems across ALL programming languages.
    - default: For general questions, explanations, or reasoning.

    ### Available Tools:
    - **codingTool** (from tools/codingTool.ts): For solving programming problems across ALL programming languages.
    - **default**: For general questions, explanations, or reasoning.

    Instructions:
    1. Always analyze the user's question carefully.
    2. If the question is related to coding, debugging, or requires generating code, use **codingTool**.
    3. Otherwise, use **default**.
    4. If context is provided, make sure to use it to enhance/improve your answer.

    ${contextText ? "Context:\n{context}\n" : ""}

    User Question:
    {question}

    Your Task:
    - Decide which tool to call (codingTool or defaultTool).
    - Provide the best possible answer.

    Decide the correct tool and provide the best possible answer.
    Answer:
  `);

  const reply = await llm.invoke(
    await prompt.format({ context: contextText, question: message })
  );

  await addToVectorStore(reply);
  return { reply, source };
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
initializeKnowledgeBase().finally(() =>
  app.listen(PORT, () =>
    logger.info(`Server running at http://localhost:${PORT}`)
  )
);
