module.exports = {
  root: true,
  env: {
    node: true,
    es2021: true,
  },
  extends: [
    'airbnb-typescript/base',
    'eslint:recommended',
    'plugin:@typescript-eslint/recommended-requiring-type-checking',
    'plugin:import/errors',
    'plugin:import/typescript',
    'prettier',
  ],
  parser: '@typescript-eslint/parser',
  parserOptions: {
    project: './tsconfig.json',
    tsconfigRootDir: __dirname,
  },
  plugins: ['@typescript-eslint', 'import'],
  rules: {
    'no-console': ['error', { allow: ['warn', 'error'] }],
    '@typescript-eslint/naming-convention': 'warn',
    'no-unused-vars': 'off',
    '@typescript-eslint/no-unused-vars': ['warn', { argsIgnorePattern: '^_' }],
    '@typescript-eslint/lines-between-class-members': 'off',
    '@typescript-eslint/restrict-template-expressions': 'error',
    '@typescript-eslint/no-use-before-define': ['error', { functions: false, classes: false }],
    'import/no-unresolved': 'error',
    'import/no-extraneous-dependencies': ['error', { devDependencies: true }],
  },
  settings: {
    'import/resolver': {
      typescript: {
        alwaysTryTypes: true,
      },
    },
  },
  overrides: [
    {
      files: ['*.js'],
      rules: {
        '@typescript-eslint/no-var-requires': 'off',
      },
    },
  ],
  ignorePatterns: ['out', 'dist', 'server', '**/*.d.ts'],
};
