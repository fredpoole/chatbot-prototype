#!/usr/bin/env python3
"""
Complexity Analysis Script
Calculates TTR, Lexical Density, Mean T-unit length, and Clauses per T-unit
for language learning transcripts.
"""

import re
import sys
from collections import Counter
from typing import List, Tuple

try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.tag import pos_tag
    from nltk.chunk import ne_chunk
    from nltk.tree import Tree
    
    # Download required NLTK data if not present
    def ensure_nltk_data():
        """Ensure all required NLTK data is downloaded."""
        required_resources = [
            ('punkt', 'tokenizers/punkt'),
            ('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger'),
            ('maxent_ne_chunker', 'chunkers/maxent_ne_chunker'),
            ('words', 'corpora/words'),
        ]
        
        # Optional resources (nice to have but not critical)
        optional_resources = [
            ('punkt_tab', 'tokenizers/punkt_tab'),
        ]
        
        missing_required = []
        missing_optional = []
        
        # Check required resources
        for resource_name, resource_path in required_resources:
            try:
                nltk.data.find(resource_path)
            except LookupError:
                missing_required.append((resource_name, resource_path))
        
        # Check optional resources
        for resource_name, resource_path in optional_resources:
            try:
                nltk.data.find(resource_path)
            except LookupError:
                missing_optional.append((resource_name, resource_path))
        
        # Only try to download if something is missing
        if missing_required or missing_optional:
            import io
            import ssl
            from contextlib import redirect_stderr
            
            # Try to download with SSL workaround if needed
            ssl_context = None
            try:
                # Create an unverified SSL context as fallback
                ssl_context = ssl._create_unverified_context()
            except AttributeError:
                pass  # Python version doesn't support this
            
            for resource_name, resource_path in missing_required:
                # Try downloading
                download_success = False
                try:
                    # Capture NLTK's stderr output
                    stderr_capture = io.StringIO()
                    with redirect_stderr(stderr_capture):
                        # Try normal download first
                        try:
                            nltk.download(resource_name, quiet=True)
                        except Exception:
                            # If that fails and we have SSL context, try with unverified SSL
                            if ssl_context:
                                import urllib.request
                                original_opener = urllib.request.urlopen
                                try:
                                    urllib.request.urlopen = lambda url, *args, **kwargs: original_opener(
                                        url, *args, context=ssl_context, **kwargs
                                    )
                                    nltk.download(resource_name, quiet=True)
                                finally:
                                    urllib.request.urlopen = original_opener
                            else:
                                raise
                    
                    # Check if it's now available
                    try:
                        nltk.data.find(resource_path)
                        download_success = True
                        # Success - resource is now available
                    except LookupError:
                        pass  # Will handle error below
                        
                except Exception as e:
                    error_msg = str(e)
                    error_output = stderr_capture.getvalue() if 'stderr_capture' in locals() else str(e)
                    
                    if not download_success:
                        # Download failed - check for SSL errors
                        if 'SSL' in error_output or 'CERTIFICATE' in error_output or 'SSL' in error_msg:
                            print(f"Warning: Could not download {resource_name} due to SSL certificate issue.", file=sys.stderr)
                            print(f"  To fix: Run '/Applications/Python\\ 3.12/Install\\ Certificates.command'", file=sys.stderr)
                            print(f"  Then: python3 -m nltk.downloader {resource_name}", file=sys.stderr)
                        else:
                            print(f"Warning: Could not download {resource_name}. Run: python3 -m nltk.downloader {resource_name}", file=sys.stderr)
            
            # Handle optional resources (silently)
            for resource_name, resource_path in missing_optional:
                try:
                    stderr_capture = io.StringIO()
                    with redirect_stderr(stderr_capture):
                        nltk.download(resource_name, quiet=True)
                    try:
                        nltk.data.find(resource_path)
                    except LookupError:
                        pass  # Optional resource failed - that's OK
                except Exception:
                    pass  # Optional resource failed - that's OK
        
        # Final check: verify required resources are available
        still_missing = []
        for resource_name, resource_path in required_resources:
            try:
                nltk.data.find(resource_path)
            except LookupError:
                still_missing.append(resource_name)
        
        if still_missing:
            print(f"ERROR: Required NLTK resources missing: {', '.join(still_missing)}", file=sys.stderr)
            print("\nTo fix this, try one of these options:", file=sys.stderr)
            print("\nOption 1 (Recommended): Fix SSL certificates, then download:", file=sys.stderr)
            print("  /Applications/Python\\ 3.12/Install\\ Certificates.command", file=sys.stderr)
            print(f"  python3 -m nltk.downloader {' '.join(still_missing)}", file=sys.stderr)
            print("\nOption 2: Use the SSL workaround script:", file=sys.stderr)
            print("  python3 download_nltk_resources.py", file=sys.stderr)
            print("\nOption 3: Download manually (after fixing SSL):", file=sys.stderr)
            print(f"  python3 -m nltk.downloader {' '.join(still_missing)}", file=sys.stderr)
            # Don't exit - let the script try to run and fail gracefully if needed
    
    # Ensure NLTK data is available (but don't block if download fails)
    try:
        ensure_nltk_data()
    except Exception as e:
        print(f"Warning: NLTK setup encountered an error: {e}", file=sys.stderr)
        print("The script will attempt to continue, but may fail if resources are missing.", file=sys.stderr)
    
except ImportError:
    print("ERROR: NLTK is required. Install with: pip install nltk", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"ERROR: NLTK setup failed: {e}", file=sys.stderr)
    print("Try running: python -m nltk.downloader all", file=sys.stderr)
    sys.exit(1)


# Lexical POS tags (nouns, verbs, adjectives, adverbs)
LEXICAL_TAGS = {'NN', 'NNS', 'NNP', 'NNPS',  # Nouns
                'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',  # Verbs
                'JJ', 'JJR', 'JJS',  # Adjectives
                'RB', 'RBR', 'RBS', 'WRB'}  # Adverbs


def extract_user_utterances(transcript: str) -> List[str]:
    """Extract user utterances from transcript (lines starting with 'You:' or 'You ')."""
    lines = transcript.split('\n')
    user_utterances = []
    for line in lines:
        line = line.strip()
        # Match lines starting with "You:" or "You " (case insensitive)
        if re.match(r'^You:?\s+', line, re.IGNORECASE):
            # Remove the "You:" or "You " prefix
            utterance = re.sub(r'^You:?\s+', '', line, flags=re.IGNORECASE)
            if utterance:
                user_utterances.append(utterance)
    return user_utterances


def tokenize_and_tag(text: str) -> List[Tuple[str, str]]:
    """Tokenize text and return list of (word, POS_tag) tuples."""
    tokens = word_tokenize(text.lower())
    tagged = pos_tag(tokens)
    return tagged


def count_lexical_words(tagged_words: List[Tuple[str, str]]) -> int:
    """Count lexical words (nouns, verbs, adjectives, adverbs)."""
    return sum(1 for word, tag in tagged_words if tag in LEXICAL_TAGS)


def identify_tunits(sentences: List[str]) -> List[str]:
    """
    Identify T-units (main clauses with all subordinate clauses).
    For simplicity, we treat each sentence as potentially containing multiple T-units.
    We split on common subordinating conjunctions.
    """
    tunits = []
    subordinating_conjunctions = [
        'because', 'since', 'although', 'though', 'while', 'when', 'if', 'unless',
        'until', 'after', 'before', 'as', 'that', 'which', 'who', 'whom', 'whose',
        'where', 'why', 'how', 'what', 'whether', 'so that', 'in order that'
    ]
    
    for sentence in sentences:
        # Simple heuristic: split on subordinating conjunctions
        # This is a simplified approach; a full parser would be more accurate
        sentence_lower = sentence.lower()
        parts = [sentence]
        
        for conj in subordinating_conjunctions:
            new_parts = []
            for part in parts:
                # Split on conjunction (with word boundaries)
                splits = re.split(rf'\b{re.escape(conj)}\b', part, flags=re.IGNORECASE)
                if len(splits) > 1:
                    # First part is main clause, rest are subordinate
                    new_parts.append(splits[0].strip())
                    for sub in splits[1:]:
                        if sub.strip():
                            new_parts.append(conj + ' ' + sub.strip())
                else:
                    new_parts.append(part)
            parts = new_parts
        
        # Filter out empty parts
        tunits.extend([p for p in parts if p.strip()])
    
    # If no T-units found, treat each sentence as one T-unit
    if not tunits and sentences:
        tunits = sentences
    
    return tunits


def count_clauses(tunit: str) -> int:
    """
    Count clauses in a T-unit.
    A clause has a subject and a predicate (verb).
    This is a simplified count based on verbs.
    """
    tagged = tokenize_and_tag(tunit)
    # Count main verbs (VB, VBD, VBG, VBN, VBP, VBZ)
    verbs = [tag for word, tag in tagged if tag.startswith('VB')]
    # Rough estimate: number of verbs approximates number of clauses
    # (this is simplified; a full parser would be more accurate)
    return max(1, len(verbs))  # At least 1 clause per T-unit


def calculate_ttr(user_text: str) -> Tuple[float, dict]:
    """Calculate Type-Token Ratio (TTR)."""
    # Tokenize to words (lowercase for consistency)
    words = word_tokenize(user_text.lower())
    # Remove punctuation-only tokens
    words = [w for w in words if re.match(r'^[a-z]', w)]
    
    total_tokens = len(words)
    unique_types = len(set(words))
    
    ttr = unique_types / total_tokens if total_tokens > 0 else 0.0
    
    details = {
        'total_tokens': total_tokens,
        'unique_types': unique_types,
        'ttr': ttr
    }
    
    return ttr, details


def calculate_lexical_density(user_text: str) -> Tuple[float, dict]:
    """Calculate Lexical Density (lexical words / total words)."""
    tagged = tokenize_and_tag(user_text)
    # Filter out punctuation
    tagged = [(w, t) for w, t in tagged if re.match(r'^[a-z]', w)]
    
    total_words = len(tagged)
    lexical_words = count_lexical_words(tagged)
    
    lexical_density = lexical_words / total_words if total_words > 0 else 0.0
    
    details = {
        'total_words': total_words,
        'lexical_words': lexical_words,
        'lexical_density': lexical_density
    }
    
    return lexical_density, details


def calculate_mean_tunit_length(user_text: str) -> Tuple[float, dict]:
    """Calculate Mean Length of T-unit in words."""
    sentences = sent_tokenize(user_text)
    tunits = identify_tunits(sentences)
    
    if not tunits:
        return 0.0, {'tunits': [], 'tunit_lengths': [], 'mean_length': 0.0}
    
    tunit_lengths = []
    for tunit in tunits:
        words = word_tokenize(tunit.lower())
        words = [w for w in words if re.match(r'^[a-z]', w)]
        tunit_lengths.append(len(words))
    
    mean_length = sum(tunit_lengths) / len(tunit_lengths) if tunit_lengths else 0.0
    
    details = {
        'tunits': tunits,
        'tunit_lengths': tunit_lengths,
        'mean_length': mean_length
    }
    
    return mean_length, details


def calculate_clauses_per_tunit(user_text: str) -> Tuple[float, dict]:
    """Calculate average clauses per T-unit."""
    sentences = sent_tokenize(user_text)
    tunits = identify_tunits(sentences)
    
    if not tunits:
        return 0.0, {'tunits': [], 'clause_counts': [], 'mean_clauses': 0.0}
    
    clause_counts = []
    for tunit in tunits:
        clause_count = count_clauses(tunit)
        clause_counts.append(clause_count)
    
    mean_clauses = sum(clause_counts) / len(clause_counts) if clause_counts else 0.0
    
    details = {
        'tunits': tunits,
        'clause_counts': clause_counts,
        'mean_clauses': mean_clauses
    }
    
    return mean_clauses, details


def analyze_complexity(transcript: str, verbose: bool = True) -> dict:
    """Main analysis function."""
    user_utterances = extract_user_utterances(transcript)
    
    if not user_utterances:
        print("WARNING: No user utterances found in transcript.")
        print("Looking for lines starting with 'You:' or 'You '")
        return {}
    
    # Combine all user utterances
    user_text = ' '.join(user_utterances)
    
    if verbose:
        print("=" * 70)
        print("COMPLEXITY ANALYSIS")
        print("=" * 70)
        print(f"\nFound {len(user_utterances)} user utterance(s):")
        for i, utt in enumerate(user_utterances, 1):
            print(f"  {i}. {utt}")
        print(f"\nCombined text: {user_text[:100]}{'...' if len(user_text) > 100 else ''}")
        print("\n" + "=" * 70)
    
    # Calculate metrics
    ttr, ttr_details = calculate_ttr(user_text)
    ld, ld_details = calculate_lexical_density(user_text)
    mtl, mtl_details = calculate_mean_tunit_length(user_text)
    cpt, cpt_details = calculate_clauses_per_tunit(user_text)
    
    results = {
        'ttr': ttr,
        'lexical_density': ld,
        'mean_tunit_length': mtl,
        'clauses_per_tunit': cpt,
        'details': {
            'ttr': ttr_details,
            'lexical_density': ld_details,
            'mean_tunit_length': mtl_details,
            'clauses_per_tunit': cpt_details
        }
    }
    
    if verbose:
        print("\n1. TYPE-TOKEN RATIO (TTR)")
        print("-" * 70)
        print(f"   Total tokens (words): {ttr_details['total_tokens']}")
        print(f"   Unique types (unique words): {ttr_details['unique_types']}")
        print(f"   TTR = unique_types / total_tokens")
        print(f"   TTR = {ttr_details['unique_types']} / {ttr_details['total_tokens']}")
        print(f"   TTR = {ttr:.3f}")
        
        print("\n2. LEXICAL DENSITY")
        print("-" * 70)
        print(f"   Total words: {ld_details['total_words']}")
        print(f"   Lexical words (nouns, verbs, adjectives, adverbs): {ld_details['lexical_words']}")
        print(f"   Lexical Density = lexical_words / total_words")
        print(f"   Lexical Density = {ld_details['lexical_words']} / {ld_details['total_words']}")
        print(f"   Lexical Density = {ld:.3f}")
        
        print("\n3. MEAN LENGTH OF T-UNIT")
        print("-" * 70)
        print(f"   Number of T-units: {len(mtl_details['tunits'])}")
        if mtl_details['tunits']:
            for i, (tunit, length) in enumerate(zip(mtl_details['tunits'], mtl_details['tunit_lengths']), 1):
                print(f"   T-unit {i}: {length} words - \"{tunit[:60]}{'...' if len(tunit) > 60 else ''}\"")
        print(f"   Mean Length = sum(lengths) / number_of_tunits")
        if mtl_details['tunit_lengths']:
            print(f"   Mean Length = {sum(mtl_details['tunit_lengths'])} / {len(mtl_details['tunit_lengths'])}")
        print(f"   Mean Length = {mtl:.2f} words")
        
        print("\n4. CLAUSES PER T-UNIT")
        print("-" * 70)
        print(f"   Number of T-units: {len(cpt_details['tunits'])}")
        if cpt_details['tunits']:
            for i, (tunit, clauses) in enumerate(zip(cpt_details['tunits'], cpt_details['clause_counts']), 1):
                print(f"   T-unit {i}: {clauses} clause(s) - \"{tunit[:60]}{'...' if len(tunit) > 60 else ''}\"")
        print(f"   Mean Clauses = sum(clause_counts) / number_of_tunits")
        if cpt_details['clause_counts']:
            print(f"   Mean Clauses = {sum(cpt_details['clause_counts'])} / {len(cpt_details['clause_counts'])}")
        print(f"   Mean Clauses = {cpt:.2f}")
        
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Type-Token Ratio (TTR):           {ttr:.3f}")
        print(f"Lexical Density:                  {ld:.3f}")
        print(f"Mean Length of T-unit:            {mtl:.2f} words")
        print(f"Clauses per T-unit:               {cpt:.2f}")
        print("=" * 70)
    
    return results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze language complexity metrics from a transcript'
    )
    parser.add_argument(
        'input',
        nargs='?',
        help='Transcript text file (or read from stdin)'
    )
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Quiet mode (only output JSON)'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )
    
    args = parser.parse_args()
    
    # Read transcript
    if args.input:
        try:
            with open(args.input, 'r', encoding='utf-8') as f:
                transcript = f.read()
        except FileNotFoundError:
            print(f"ERROR: File '{args.input}' not found.", file=sys.stderr)
            sys.exit(1)
    else:
        # Read from stdin
        transcript = sys.stdin.read()
    
    if not transcript.strip():
        print("ERROR: No transcript provided.", file=sys.stderr)
        sys.exit(1)
    
    # Analyze
    results = analyze_complexity(transcript, verbose=not args.quiet)
    
    # Output JSON if requested
    if args.json or args.quiet:
        import json
        output = {
            'ttr': results.get('ttr', 0),
            'lexical_density': results.get('lexical_density', 0),
            'mean_tunit_length': results.get('mean_tunit_length', 0),
            'clauses_per_tunit': results.get('clauses_per_tunit', 0)
        }
        print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()

