import katex from 'katex'
import { useMemo } from 'react'

/** KaTeX-rendered math. `block` renders display style. */
export function Tex({ children, block = false }: { children: string; block?: boolean }) {
  const html = useMemo(() => katex.renderToString(children, { displayMode: block, throwOnError: false, strict: 'ignore' }), [children, block])
  return block ? <div dangerouslySetInnerHTML={{ __html: html }} /> : <span dangerouslySetInnerHTML={{ __html: html }} />
}
