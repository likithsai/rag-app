import { CodingTool } from "./codingTool";
import { DefaultTool } from "./defaultTool";
import { BaseTool } from "../classes/baseTool";

export const tools: Record<string, BaseTool> = {
  codingTool: new CodingTool(),
  default: new DefaultTool(),
};
