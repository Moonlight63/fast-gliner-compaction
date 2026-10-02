import { GlinerClient, type GlinerClientOptions } from './client.js';
import { compact } from './compact.js';
import type { CompactOptions, CompactResult, Message } from './types.js';

export type CompactMessagesOptions = CompactOptions & GlinerClientOptions;

/** `compact` with a `GlinerClient` built from the options (server from `FGC_SERVER_URL` by default). */
export function compactMessages(
  messages: readonly Message[],
  options: CompactMessagesOptions = {},
): Promise<CompactResult> {
  return compact(messages, new GlinerClient(options), options);
}
