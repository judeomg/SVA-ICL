from tree_sitter import Language

Language.build_library(

  # Store the library in the `build` directory
  'build/my-languages.so',

  # Include one or more languages
  [
    'C:/Data/PycharmProject/treesitter/tree-sitter-c',
    'C:/Data/PycharmProject/treesitter/tree-sitter-cpp'
    # 'treesitter/tree-sitter-java',
    # 'treesitter/tree-sitter-python',
    # 'treesitter/tree-sitter-cpp',
  ]
)
