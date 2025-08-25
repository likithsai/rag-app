import { ToolInput } from "../interfaces/tools";

export abstract class BaseTool {
  abstract name: string;
  abstract description: string;
  abstract run(input: ToolInput): Promise<string> | string;
}
