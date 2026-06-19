// KaTeX auto-render не имеет типов в пакете — минимальная декларация (рендерит $…$/$$…$$ в элементе).
declare module "katex/contrib/auto-render" {
  interface Delim { left: string; right: string; display: boolean }
  interface Opts { delimiters?: Delim[]; throwOnError?: boolean; [k: string]: unknown }
  const renderMathInElement: (el: HTMLElement, options?: Opts) => void
  export default renderMathInElement
}
