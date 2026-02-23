export const STOPWORDS = [
  "help",
  "please",
  "argh",
  "oh",
  "god",
  "um",
  "uh",
  "like",
  "just",
  "so",
  "very",
  "really",
  "can",
  "you",
  "tell",
  "me",
  "about",
  "i",
  "think",
  "maybe",
  "what",
  "is",
  "the",
  "a",
  "an",
  "in",
  "on",
  "at",
  "to",
  "for",
  "of",
  "with",
  "by"
];

/**
 * Removes common stopwords from a text string.
 * @param text The input text to clean.
 * @returns The cleaned text with stopwords removed.
 */
export const removeStopwords = (text: string): string => {
  if (!text) return "";
  
  // Create a regex pattern to match whole words, case-insensitive
  const words = text.split(/\s+/);
  const filteredWords = words.filter(word => {
    const cleanWord = word.toLowerCase().replace(/[^a-z0-9]/g, ""); // basic punctuation strip
    return !STOPWORDS.includes(cleanWord);
  });
  
  return filteredWords.join(" ");
};
