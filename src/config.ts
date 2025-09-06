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
  CHROMA_HOST: process.env.CHROMA_HOST || "chroma",
  CHROMA_PORT: Number(process.env.CHROMA_PORT || 8080),
  SUPPORTED_FORMATS: (
    process.env.SUPPORTED_FORMATS || ".pdf,.txt,.docx,.csv,.html,.md"
  )
    .split(",")
    .map((f) => f.trim().toLowerCase()),
};
