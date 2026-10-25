// The Expressive Code plugin that marks an output block: a fence with the flag
// `output`, such as ```text output, gets the class gs-output on its group, which
// src/styles/custom.css joins to the code block above it.
import { addClassName } from '@astrojs/starlight/expressive-code/hast'

export const outputBlocks = {
  name: 'output-blocks',
  hooks: {
    postprocessRenderedBlockGroup({ renderedGroupContents, renderData }) {
      if (renderedGroupContents.some(({ codeBlock }) => codeBlock.metaOptions.getBoolean('output'))) {
        addClassName(renderData.groupAst, 'gs-output')
      }
    },
  },
}
