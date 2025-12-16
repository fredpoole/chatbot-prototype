# Complexity Analysis Script

This script (`analyze_complexity.py`) calculates language complexity metrics from conversation transcripts.

## Installation

First, install the required dependency:

```bash
pip install nltk
```

### Fix SSL Certificate Issues (macOS)

If you encounter SSL certificate errors when downloading NLTK resources, fix it first:

**Option 1: Install Python Certificates (Recommended)**
```bash
/Applications/Python\ 3.12/Install\ Certificates.command
```

Or find your Python version:
```bash
# Find your Python installation
python3 --version
# Then run the Install Certificates script for that version
# Example: /Applications/Python\ 3.11/Install\ Certificates.command
```

**Option 2: Download Resources Manually**
If SSL issues persist, you can manually download the NLTK data files and place them in the NLTK data directory. The script will detect them automatically.

### Download NLTK Data

After fixing SSL issues, download the required NLTK data:

```bash
python3 -m nltk.downloader punkt punkt_tab averaged_perceptron_tagger maxent_ne_chunker words
```

Note: `punkt_tab` is optional but recommended for newer NLTK versions.

Or download all NLTK data (larger download):

```bash
python3 -m nltk.downloader all
```

The script will attempt to download missing resources automatically, but manual download is recommended.

**Troubleshooting**: If you see SSL errors, the script will provide helpful error messages and continue if optional resources fail. Required resources must be downloaded manually if automatic download fails.

## Usage

### Basic Usage

Analyze a transcript file:

```bash
python3 analyze_complexity.py transcript.txt
```

Or pipe transcript from stdin:

```bash
cat transcript.txt | python3 analyze_complexity.py
```

### Options

- `-q, --quiet`: Quiet mode (only output JSON)
- `--json`: Output results as JSON format

Example with JSON output:

```bash
python3 analyze_complexity.py transcript.txt --json
```

## Metrics Calculated

The script calculates four complexity metrics:

### 1. Type-Token Ratio (TTR)
- **Formula**: unique_words / total_words
- **Range**: 0.0 to 1.0
- **Higher values** indicate more vocabulary diversity

### 2. Lexical Density
- **Formula**: lexical_words / total_words
- **Range**: 0.0 to 1.0
- **Lexical words**: nouns, verbs, adjectives, adverbs
- **Higher values** indicate more content words vs. function words

### 3. Mean Length of T-unit
- **Formula**: average words per T-unit
- **T-unit**: A main clause with all its subordinate clauses
- **Higher values** indicate more complex sentence structure

### 4. Clauses per T-unit
- **Formula**: average number of clauses per T-unit
- **Clause**: A unit with a subject and predicate (verb)
- **Higher values** indicate more subordination and complexity

## Transcript Format

The script looks for user utterances starting with:
- `You: ...`
- `You ...`

Example transcript format:

```
You: Hello, I want to rent an apartment.
Agent: That's great! What's your budget?
You: I think maybe around one thousand dollars per month would be good.
```

## Example Output

```
======================================================================
COMPLEXITY ANALYSIS
======================================================================

Found 3 user utterance(s):
  1. Hello, I want to rent an apartment.
  2. I think maybe around one thousand dollars per month would be good.
  3. I need two bedrooms because I have a roommate who will live with me.

Combined text: hello , i want to rent an apartment . i think maybe around one thousand dollars per month would be good . i need two bedrooms because i have a roommate who will live with me .

======================================================================

1. TYPE-TOKEN RATIO (TTR)
----------------------------------------------------------------------
   Total tokens (words): 25
   Unique types (unique words): 22
   TTR = unique_types / total_tokens
   TTR = 22 / 25
   TTR = 0.880

2. LEXICAL DENSITY
----------------------------------------------------------------------
   Total words: 25
   Lexical words (nouns, verbs, adjectives, adverbs): 15
   Lexical Density = lexical_words / total_words
   Lexical Density = 15 / 25
   Lexical Density = 0.600

3. MEAN LENGTH OF T-UNIT
----------------------------------------------------------------------
   Number of T-units: 3
   T-unit 1: 6 words - "hello , i want to rent an apartment ."
   T-unit 2: 10 words - "i think maybe around one thousand dollars per month would be good ."
   T-unit 3: 12 words - "i need two bedrooms because i have a roommate who will live with me ."
   Mean Length = sum(lengths) / number_of_tunits
   Mean Length = 28 / 3
   Mean Length = 9.33 words

4. CLAUSES PER T-UNIT
----------------------------------------------------------------------
   Number of T-units: 3
   T-unit 1: 1 clause(s) - "hello , i want to rent an apartment ."
   T-unit 2: 1 clause(s) - "i think maybe around one thousand dollars per month would be good ."
   T-unit 3: 2 clause(s) - "i need two bedrooms because i have a roommate who will live with me ."
   Mean Clauses = sum(clause_counts) / number_of_tunits
   Mean Clauses = 4 / 3
   Mean Clauses = 1.33

======================================================================
SUMMARY
======================================================================
Type-Token Ratio (TTR):           0.880
Lexical Density:                  0.600
Mean Length of T-unit:            9.33 words
Clauses per T-unit:               1.33
======================================================================
```

## Notes

- The script focuses on **user utterances only** (ignores agent responses)
- T-unit identification uses heuristics based on subordinating conjunctions
- Clause counting is based on verb detection (simplified approach)
- For production use, consider using more sophisticated parsers (e.g., spaCy with dependency parsing)

## Integration

You can use this script to verify the calculations made by the main server's analysis endpoint, or run it independently to analyze transcripts.

