import { buildDecideRequest, parseDecideResponse } from './request.js';
import type { DecideResponse, DecisionQuestions, Decider } from './types.js';

export interface GlinerClientOptions {
  /** Defaults to `process.env.FGC_SERVER_URL`, then `http://127.0.0.1:8765`. */
  serverUrl?: string;
  /** Bearer token; defaults to `process.env.FGC_TOKEN`. */
  token?: string;
  /** Server model name; defaults to `gliner-decide`. */
  model?: string;
  /** Defaults to the global `fetch`. */
  fetch?: typeof fetch;
}

/** Asks the decision server over HTTP with the global `fetch` (or an injected one). */
export class GlinerClient implements Decider {
  private readonly serverUrl: string | undefined;
  private readonly token: string | undefined;
  private readonly model: string | undefined;
  private readonly fetcher: typeof fetch;

  constructor(options: GlinerClientOptions = {}) {
    this.serverUrl = options.serverUrl ?? process.env.FGC_SERVER_URL;
    this.token = options.token ?? process.env.FGC_TOKEN;
    this.model = options.model;
    this.fetcher = options.fetch ?? fetch;
  }

  async decide(items: readonly string[], questions: DecisionQuestions): Promise<DecideResponse> {
    const request = buildDecideRequest(
      { serverUrl: this.serverUrl, token: this.token, model: this.model },
      items,
      questions,
    );
    const response = await this.fetcher(request.url, {
      method: request.method,
      headers: request.headers,
      body: request.body,
    });
    return parseDecideResponse(
      response.status,
      response.ok,
      await response.text(),
      items.length,
      questions,
    );
  }
}
