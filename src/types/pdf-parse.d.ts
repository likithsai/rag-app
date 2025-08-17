declare module "pdf-parse" {
  interface PDFInfo {
    PDFFormatVersion: string;
    IsAcroFormPresent: boolean;
    IsXFAPresent: boolean;
    numpages: number;
    numrender: number;
    info: Record<string, any>;
    metadata?: any;
    text: string;
    version: string;
  }

  function pdf(dataBuffer: Buffer, options?: any): Promise<PDFInfo>;

  export = pdf;
}
