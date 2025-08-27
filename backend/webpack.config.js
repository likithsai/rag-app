// webpack.config.js
const path = require("path");
const nodeExternals = require("webpack-node-externals");
const TerserPlugin = require("terser-webpack-plugin");
const WebpackObfuscator = require('webpack-obfuscator');

module.exports = {
    mode: "production", // ensures minification
    target: "node",     // build for Node.js
    entry: "./src/index.ts", // your entry file (change if needed)
    output: {
        path: path.resolve(__dirname, "dist"),
        filename: "server.min.js",
        clean: true,
    },
    resolve: {
        extensions: [".ts", ".js", ".json", ".tsx"],
    },
    module: {
        rules: [
            {
                test: /\.tsx?$/,
                use: 'ts-loader',
                exclude: /node_modules/,
            },
        ],
    },
    externals: [nodeExternals()], // keep node_modules out unless bundled
    optimization: {
        minimize: true,
        minimizer: [
            new TerserPlugin({
                terserOptions: {
                    mangle: true, // rename variables/functions
                    compress: true,
                },
            }),
        ],
    },
    plugins: [
        new WebpackObfuscator(
            {
                compact: true,
                controlFlowFlattening: false,      // enable only if you really want max obfuscation (slower)
                deadCodeInjection: false,          // safe default for servers
                debugProtection: false,
                disableConsoleOutput: false,
                identifierNamesGenerator: 'hexadecimal',
                numberToExpression: true,
                renameGlobals: false,
                rotateStringArray: true,
                selfDefending: false,              // must be false for Node servers
                shuffleStringArray: true,
                simplify: true,
                splitStrings: true,
                splitStringsChunkLength: 8,
                stringArray: true,
                stringArrayCallsTransform: true,
                stringArrayEncoding: ['base64'],
                stringArrayThreshold: 0.75,
                transformObjectKeys: true,
                unicodeEscapeSequence: false,
            },
            // Exclude source maps or other files from obfuscation if needed
            ['**/*.map']
        ),
    ],
    optimization: {
        minimize: false, // let obfuscator handle transformation; set to true if you also want minification by Terser
    },
    devtool: false,
};