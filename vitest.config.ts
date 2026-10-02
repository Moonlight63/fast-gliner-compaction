import { defineConfig } from 'vitest/config';

// *.spec.ts run under vitest; *.test.ts are engine tests for `claude plugin test`.
export default defineConfig({
  test: { include: ['tests/**/*.spec.ts'] },
});
