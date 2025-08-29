// backend/src/config.ts
import dotenv from "dotenv";
import path from "path";

dotenv.config();

export const Config = {
  PORT: process.env.PORT || 3001,
  PUBLIC_FOLDER: process.env.PUBLIC_FOLDER
    ? path.resolve(process.cwd(), process.env.PUBLIC_FOLDER)
    : path.resolve(process.cwd(), "pdfs"),
  OLLAMA_BASE_URL: process.env.OLLAMA_BASE_URL || "http://localhost",
  OLLAMA_MODEL: process.env.OLLAMA_MODEL || "llama3.1",
  OLLAMA_PORT: process.env.OLLAMA_PORT || 11434,
  EMBEDDING_MODEL: process.env.EMBEDDING_MODEL || "nomic-embed-text",
  VECTOR_STORE_PATH: process.env.VECTOR_STORE_PATH
    ? path.resolve(process.cwd(), process.env.VECTOR_STORE_PATH)
    : path.resolve(process.cwd(), "vector_store"),
  SUPPORTED_FORMATS: (
    process.env.SUPPORTED_FORMATS || ".pdf,.txt,.docx,.csv,.html,.md"
  )
    .split(",")
    .map((f) => f.trim().toLowerCase()),
};
