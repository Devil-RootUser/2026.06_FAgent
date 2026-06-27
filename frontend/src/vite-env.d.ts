interface ImportMetaEnv {
  readonly VITE_API_BASE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

declare module "react" {
  export const useEffect: any;
  export const useMemo: any;
  export function useState<T = any>(initial?: T): [T, (value: T | ((prev: T) => T)) => void];
  export type DependencyList = any[];
  const React: any;
  export default React;
}

declare module "react-dom/client" {
  export function createRoot(element: Element): { render(node: any): void };
}

declare module "react/jsx-runtime" {
  export const jsx: any;
  export const jsxs: any;
  export const Fragment: any;
}

declare namespace JSX {
  interface IntrinsicElements {
    [elemName: string]: any;
  }
}

