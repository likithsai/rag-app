export interface ToolInput {
  question: string;
  context?: string;
}

export abstract class Tool {
  abstract name: string;
  abstract description: string;
  abstract run(input: ToolInput): Promise<string> | string;
}
