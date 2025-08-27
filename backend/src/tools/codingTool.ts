// codingTool.ts
import { PromptTemplate } from "@langchain/core/prompts";
import { LLMChain } from "langchain/chains";
import { Ollama } from "@langchain/community/llms/ollama";
import { BaseTool } from "../classes/baseTool";
import { ToolInput } from "../interfaces/tools";

export class CodingTool implements BaseTool {
  name = "codingTool";
  description = "Solves programming problems across all programming languages.";

  private chain: LLMChain;

  constructor(model = "llama3.1") {
    const llm = new Ollama({
      model,
      temperature: 0.7,
    });

    const codingPrompt = PromptTemplate.fromTemplate(`
        You are an expert coding assistant.

        {context}

        **Rules:**
        1. Analyze the user's question carefully.
        2. Answer coding or programming questions with complete, working code.
        3. Support all programming languages.
        4. Add explanations before the code.
        5. Include inline comments for clarity.

        **User Question:**
        {question}

        Answers:
    `);

    this.chain = new LLMChain({
      llm,
      prompt: codingPrompt,
    });
  }

  async run(input: ToolInput): Promise<string> {
    const result = await this.chain.call({
      question: input.question,
      context: input.context ?? "",
    });

    // In latest LangChain, LLMChain.call returns { text: string }
    return result?.text ?? JSON.stringify(result);
  }
}
