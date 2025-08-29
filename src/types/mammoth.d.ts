declare module "mammoth" {
  export function extractRawText(options: {
    path: string;
  }): Promise<{ value: string }>;
}
