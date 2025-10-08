import React, { createElement, forwardRef } from "react"
import defaultAttributes from "./defaultAttributes"
// Simple utility to merge class names
function mergeClasses(...classes: (string | undefined)[]) {
  return classes.filter(Boolean).join(" ");
}

export interface IconNode extends Array<[string, Record<string, any>]> {}

export interface LucideProps extends React.SVGProps<SVGSVGElement> {
  color?: string
  size?: number | string
  strokeWidth?: number | string
  absoluteStrokeWidth?: boolean
  className?: string
  children?: React.ReactNode
  iconNode?: IconNode
}

/**
 * Low-level Lucide Icon component
 * (Normally you import specific icons instead, e.g. `import { Home } from "lucide-react"`)
 */
const Icon = forwardRef<SVGSVGElement, LucideProps>(
  (
    {
      color = "currentColor",
      size = 24,
      strokeWidth = 2,
      absoluteStrokeWidth,
      className = "",
      children,
      iconNode = [], // ✅ default to empty array so .map() is safe
      ...rest
    },
    ref
  ) => {
    return createElement(
      "svg",
      {
        ref,
        ...defaultAttributes,
        width: size,
        height: size,
        stroke: color,
        strokeWidth: absoluteStrokeWidth
          ? (Number(strokeWidth) * 24) / Number(size)
          : strokeWidth,
        className: mergeClasses("lucide", className),
        ...rest,
      },
      [
        // ✅ safe map (won’t crash if iconNode is missing)
        ...iconNode.map(([tag, attrs], i) => createElement(tag, { key: i, ...attrs })),
        ...(Array.isArray(children) ? children : children ? [children] : []),
      ]
    )
  }
)

Icon.displayName = "Icon"

export default Icon
