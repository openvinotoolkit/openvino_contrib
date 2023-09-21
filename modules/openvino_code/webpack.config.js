// @ts-check
const path = require('path');
const CopyPlugin = require('copy-webpack-plugin');
const TerserPlugin = require('terser-webpack-plugin');
const { BundleAnalyzerPlugin } = require('webpack-bundle-analyzer');

/** @typedef {import('webpack').Configuration} WebpackConfig **/

/** @type WebpackConfig */
const config = {
  target: 'node', // vscode extensions run in a Node.js-context 📖 -> https://webpack.js.org/configuration/node/

  entry: {
    extension: './src/extension.ts',
  }, // the entry point of this extension, 📖 -> https://webpack.js.org/configuration/entry-context/
  output: {
    // the bundle is stored in the 'dist' folder (check package.json), 📖 -> https://webpack.js.org/configuration/output/
    path: path.resolve(__dirname, 'out'),
    filename: `[name].js`,
    libraryTarget: 'commonjs2',
    devtoolModuleFilenameTemplate: '../[resource-path]',
  },
  node: {
    __dirname: false, // leave the __dirname behavior intact
  },
  devtool: 'source-map',
  externals: {
    vscode: 'commonjs vscode', // the vscode-module is created on-the-fly and must be excluded. Add other modules that cannot be webpack'ed, 📖 -> https://webpack.js.org/configuration/externals/
  },
  resolve: {
    // support reading TypeScript and JavaScript files, 📖 -> https://github.com/TypeStrong/ts-loader
    extensions: ['.ts', '.js'],
    alias: {
      '@shared': path.resolve(__dirname, 'shared'),
    },
  },
  module: {
    rules: [
      {
        test: /\.ts$/,
        exclude: /node_modules/,
        use: [
          {
            loader: 'ts-loader',
          },
        ],
      },
    ],
  },
  plugins: [
    new CopyPlugin({
      patterns: [
        { from: './src/docstring/docstring-template/templates', to: './doc_string/templates' },
        { from: './server', to: './server', globOptions: { gitignore: true } },
      ],
    }),
  ],
  optimization: {
    minimizer: [new TerserPlugin({ extractComments: false })],
  },
};

/**
 * @typedef {Function}
 * @param {{ analyzeBundle: boolean }} env
 * */
module.exports = (env) => {
  if (env.analyzeBundle) {
    config.plugins?.push(new BundleAnalyzerPlugin());
  }

  return [config];
};
