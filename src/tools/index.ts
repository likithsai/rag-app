import { CodingTool } from "./codingTool";
import { DefaultTool } from "./defaultTool";
import { Tool } from "./baseTool";

export const tools: Record<string, Tool> = {
  codingTool: new CodingTool(),
  default: new DefaultTool(),
};
