/**
 * @see https://prettier.io/docs/en/configuration.html
 */
const config = {
  tabWidth: 2,
  printWidth: 80,
  plugins: [
    require("prettier-plugin-sql"),
    require("@prettier/plugin-xml"),
    require("prettier-plugin-gherkin"),
  ],
};

module.exports = config;
