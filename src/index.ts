import bodyParser from "body-parser";
import cors from "cors";
import crypto from "crypto";
import csvParser from "csv-parser";
import express, { NextFunction, Request, Response } from "express";
import fs from "fs";
import { JSDOM } from "jsdom";
import mammoth from "mammoth";
import path from "path";
import pdfParse from "pdf-parse";
import winston from "winston";

import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";
import { Ollama } from "@langchain/community/llms/ollama";
import { PromptTemplate } from "@langchain/core/prompts";
import { Document } from "langchain/document";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { ChromaClient } from "chromadb";
import { Chroma } from "@langchain/community/vectorstores/chroma";

import pkg from "../package.json";
import { Config } from "./config";
import { tools } from "./tools";

const TOP_K = 5;
const OLLAMA_SERVER = `${Config.OLLAMA_BASE_URL}:${Config.OLLAMA_PORT}`;
const CHROMA_URL = Config.CHROMA_URL;

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

// --- Vector store via Chroma
const chroma = new ChromaClient({ path: CHROMA_URL });
let collection: any = null;

// --- Helpers
const fileExists = (p: string) => fs.existsSync(p);
const isSupportedFile = (filePath: string) =>
  Config.SUPPORTED_FORMATS.includes(path.extname(filePath).toLowerCase());

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
    const embeddings = new OllamaEmbeddings({
      model: Config.EMBEDDING_MODEL, // e.g. "nomic-embed-text"
      baseUrl: OLLAMA_SERVER,
    });

    // Create or load collection
    collection = await Chroma.fromExistingCollection(embeddings, {
      collectionName: "rag-app",
      url: CHROMA_URL,
    }).catch(async () => {
      logger.info("No existing collection, creating new one...");

      const files = getAllFiles(Config.PUBLIC_FOLDER);
      if (!files.length) {
        logger.warn("No files found in knowledge base folder.");
        return null;
      }

      const allDocs = (
        await Promise.all(files.map((f) => processFile(f)))
      ).flat();

      if (!allDocs.length) {
        logger.warn("No documents processed for knowledge base.");
        return null;
      }

      const store = await Chroma.fromDocuments(allDocs, embeddings, {
        collectionName: "rag-app",
        url: CHROMA_URL,
      });

      logger.info(`Knowledge base initialized with ${allDocs.length} chunks.`);
      return store;
    });

    if (collection) {
      logger.info("✅ Chroma collection ready");
    } else {
      logger.warn("⚠️ Knowledge base not initialized");
    }
  } catch (err) {
    logger.error(`Knowledge base init failed: ${(err as Error).message}`);
  }
}

// --- Add chat replies to vector store
async function addToVectorStore(text: string, sourceName = "chat") {
  if (!collection || !text.trim()) return;

  const hash = crypto.createHash("sha256").update(text).digest("hex");

  const embedding = await new OllamaEmbeddings({
    model: Config.EMBEDDING_MODEL,
    baseUrl: OLLAMA_SERVER,
  }).embedQuery(text);

  await collection.add({
    ids: [hash],
    documents: [text],
    metadatas: [{ source: sourceName, hash }],
    embeddings: [embedding],
  });
}

// --- Chat handler
async function handleChat(message: string, useRAG = false) {
  const llm = new Ollama({
    model: Config.OLLAMA_MODEL,
    temperature: 0.7,
    baseUrl: OLLAMA_SERVER,
  });

  let contextText = "";
  if (collection && useRAG) {
    const embeddings = new OllamaEmbeddings({
      model: Config.EMBEDDING_MODEL,
      baseUrl: OLLAMA_SERVER,
    });
    const queryEmbedding = await embeddings.embedQuery(message);

    const results = await collection.query({
      queryEmbeddings: [queryEmbedding],
      nResults: TOP_K,
    });

    contextText = (results.documents[0] || []).join("\n\n");
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
// --- List files in RAG
app.get("/rag-files", async (_req, res) => {
  try {
    const files = getAllFiles(Config.PUBLIC_FOLDER);

    res.json({
      totalFiles: files.length,
      files,
    });
  } catch (err) {
    logger.error(`Failed to fetch RAG files: ${(err as Error).message}`);
    res.status(500).json({ error: "Failed to fetch RAG files" });
  }
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
  logger.error(err.stack);
  res.status(500).json({ error: err.message });
});

// --- Start server
console.log(`\n${pkg.name} v${pkg.version}`);

(async () => {
  await initializeKnowledgeBase();

  app.listen(Config.PORT, () => {
    logger.info("✅ Server started successfully");
    logger.info(`🌐 Running at: http://localhost:${Config.PORT}`);
    logger.info(
      `⚙️ Node: ${process.version}, Embeddings: ${Config.EMBEDDING_MODEL}, LLM: ${Config.OLLAMA_MODEL}`
    );
  });
})();
