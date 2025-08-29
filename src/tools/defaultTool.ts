// defaultTool.ts
import { PromptTemplate } from "@langchain/core/prompts";
import { LLMChain } from "langchain/chains";
import { Ollama } from "@langchain/community/llms/ollama";
import { BaseTool } from "../classes/baseTool";
import { ToolInput } from "../interfaces/tools";

export class DefaultTool implements BaseTool {
  name = "defaultTool";
  description =
    "General-purpose tool that returns answers directly from the LLM for coding or other questions.";

  private chain: LLMChain;

  constructor(model = "llama3.1") {
    const llm = new Ollama({
      model,
      temperature: 0.7,
    });

    const defaultPrompt = PromptTemplate.fromTemplate(`
      You are a highly capable AI assistant.

      **Instructions:**
      1. Analyze the user's question carefully.
      2. Answer coding or programming questions with complete, working code.
      3. Answer general questions clearly and concisely.
      4. Use the provided context if available to improve your answer.

      **Context:**
      {context}

      **User Question:**
      {question}

      Answers:
    `);

    this.chain = new LLMChain({
      llm,
      prompt: defaultPrompt,
    });
  }

  async run(input: ToolInput): Promise<string> {
    const result = await this.chain.call({
      question: input.question,
      context: input.context ?? "",
    });

    // Return LLM output directly
    return result?.text ?? JSON.stringify(result);
  }
}
