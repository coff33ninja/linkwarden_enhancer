"""Description generation engine that orchestrates all description sources"""

import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from enhancement.meta_description_extractor import MetaDescriptionExtractor, MetaDescriptionResult
from enhancement.ai_summarizer import AISummarizer, SummarizationResult
from enhancement.content_extractor import ContentExtractor, ContentSnippet
from utils.logging_utils import get_logger
from utils.text_utils import TextUtils

logger = get_logger(__name__)


class DescriptionSource(Enum):
    """Sources for bookmark descriptions"""
    USER_EXISTING = "user_existing"
    META_DESCRIPTION = "meta_description"
    AI_SUMMARY = "ai_summary"
    CONTENT_SNIPPET = "content_snippet"
    FALLBACK = "fallback"


@dataclass
class DescriptionCandidate:
    """A candidate description from a specific source"""
    text: str
    source: DescriptionSource
    confidence: float
    quality_score: float
    length: int
    processing_time: float
    metadata: Dict[str, Any]


@dataclass
class DescriptionGenerationResult:
    """Result from description generation process"""
    final_description: str
    source_used: DescriptionSource
    confidence: float
    quality_score: float
    processing_time: float
    candidates_evaluated: List[DescriptionCandidate]
    preservation_applied: bool
    merging_applied: bool
    validation_passed: bool
    validation_issues: List[str]


class DescriptionGenerator:
    """Main description generation engine that orchestrates all sources"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize description generator"""
        self.config = config
        self.description_config = config.get('description', {})
        
        # Initialize component extractors
        self.meta_extractor = MetaDescriptionExtractor(config)
        self.ai_summarizer = AISummarizer(config)
        self.content_extractor = ContentExtractor(config)
        
        # Description settings
        self.min_length = self.description_config.get('min_length', 50)
        self.max_length = self.description_config.get('max_length', 200)
        self.ideal_length_min = self.description_config.get('ideal_length_min', 100)
        self.ideal_length_max = self.description_config.get('ideal_length_max', 180)
        
        # Source preferences and thresholds
        self.preserve_user_descriptions = self.description_config.get('preserve_user_descriptions', True)
        self.user_description_quality_threshold = self.description_config.get('user_quality_threshold', 0.6)
        self.min_confidence_threshold = self.description_config.get('min_confidence_threshold', 0.4)
        self.quality_improvement_threshold = self.description_config.get('quality_improvement_threshold', 0.2)
        
        # Source priority and weights
        self.source_priorities = {
            DescriptionSource.USER_EXISTING: 1.0,
            DescriptionSource.META_DESCRIPTION: 0.9,
            DescriptionSource.AI_SUMMARY: 0.8,
            DescriptionSource.CONTENT_SNIPPET: 0.6,
            DescriptionSource.FALLBACK: 0.3
        }
        
        # Enable/disable sources
        self.enable_meta_extraction = self.description_config.get('enable_meta_extraction', True)
        self.enable_ai_summarization = self.description_config.get('enable_ai_summarization', True)
        self.enable_content_extraction = self.description_config.get('enable_content_extraction', True)
        
        logger.info("Description generator initialized")
    
    def generate_description(self, 
                           title: str,
                           content: str,
                           url: str,
                           existing_description: str = "",
                           html_content: str = "") -> DescriptionGenerationResult:
        """Generate the best description from all available sources"""
        
        start_time = time.time()
        candidates = []
        
        try:
            # Step 1: Evaluate existing user description
            user_candidate = self._evaluate_existing_description(existing_description)
            if user_candidate:
                candidates.append(user_candidate)
            
            # Step 2: Extract meta description
            if self.enable_meta_extraction and html_content:
                meta_candidate = self._extract_meta_description(html_content, url, title)
                if meta_candidate:
                    candidates.append(meta_candidate)
            
            # Step 3: Generate AI summary
            if self.enable_ai_summarization and content:
                ai_candidate = self._generate_ai_summary(title, content, url)
                if ai_candidate:
                    candidates.append(ai_candidate)
            
            # Step 4: Extract content snippet
            if self.enable_content_extraction and (html_content or url):
                snippet_candidate = self._extract_content_snippet(html_content, url, title)
                if snippet_candidate:
                    candidates.append(snippet_candidate)
            
            # Step 5: Create fallback if no candidates
            if not candidates:
                fallback_candidate = self._create_fallback_description(title, url)
                candidates.append(fallback_candidate)
            
            # Step 6: Select best description
            best_candidate = self._select_best_description(candidates, existing_description)
            
            # Step 7: Apply preservation and merging logic
            final_candidate, preservation_applied, merging_applied = self._apply_preservation_and_merging(
                best_candidate, user_candidate, candidates
            )
            
            # Step 8: Validate and format final description
            final_description, validation_passed, validation_issues = self._validate_and_format_description(
                final_candidate.text
            )
            
            processing_time = time.time() - start_time
            
            return DescriptionGenerationResult(
                final_description=final_description,
                source_used=final_candidate.source,
                confidence=final_candidate.confidence,
                quality_score=final_candidate.quality_score,
                processing_time=processing_time,
                candidates_evaluated=candidates,
                preservation_applied=preservation_applied,
                merging_applied=merging_applied,
                validation_passed=validation_passed,
                validation_issues=validation_issues
            )
            
        except Exception as e:
            logger.error(f"Description generation failed: {e}")
            
            # Emergency fallback
            fallback_description = f"Content titled '{title}'" if title else "Web content"
            processing_time = time.time() - start_time
            
            return DescriptionGenerationResult(
                final_description=fallback_description,
                source_used=DescriptionSource.FALLBACK,
                confidence=0.1,
                quality_score=0.2,
                processing_time=processing_time,
                candidates_evaluated=candidates,
                preservation_applied=False,
                merging_applied=False,
                validation_passed=False,
                validation_issues=[f"Generation failed: {e}"]
            )
    
    def _evaluate_existing_description(self, existing_description: str) -> Optional[DescriptionCandidate]:
        """Evaluate existing user description"""
        
        if not existing_description or not existing_description.strip():
            return None
        
        try:
            cleaned_description = existing_description.strip()
            
            # Calculate quality score for existing description
            quality_score = self._calculate_description_quality(cleaned_description, 'user_existing')
            
            # High confidence for user content (they wrote it for a reason)
            base_confidence = 0.9 if self.preserve_user_descriptions else 0.7
            
            # Adjust confidence based on quality
            confidence = base_confidence * (0.7 + quality_score * 0.3)
            
            return DescriptionCandidate(
                text=cleaned_description,
                source=DescriptionSource.USER_EXISTING,
                confidence=confidence,
                quality_score=quality_score,
                length=len(cleaned_description),
                processing_time=0.0,
                metadata={'original_length': len(existing_description)}
            )
            
        except Exception as e:
            logger.debug(f"Failed to evaluate existing description: {e}")
            return None
    
    def _extract_meta_description(self, html_content: str, url: str, title: str) -> Optional[DescriptionCandidate]:
        """Extract meta description candidate"""
        
        try:
            start_time = time.time()
            
            result = self.meta_extractor.extract_from_html(html_content, url)
            
            if not result.description:
                return None
            
            processing_time = time.time() - start_time
            
            return DescriptionCandidate(
                text=result.description,
                source=DescriptionSource.META_DESCRIPTION,
                confidence=result.confidence,
                quality_score=result.quality_score,
                length=result.length,
                processing_time=processing_time,
                metadata={
                    'meta_source': result.source,
                    'fallback_used': result.fallback_used
                }
            )
            
        except Exception as e:
            logger.debug(f"Meta description extraction failed: {e}")
            return None
    
    def _generate_ai_summary(self, title: str, content: str, url: str) -> Optional[DescriptionCandidate]:
        """Generate AI summary candidate"""
        
        try:
            # Determine appropriate length for AI summary
            ai_max_length = min(self.max_length, 180)  # AI summaries tend to be more concise
            
            result = self.ai_summarizer.summarize_content(
                title=title,
                content=content,
                url=url,
                max_length=ai_max_length
            )
            
            if not result.summary:
                return None
            
            return DescriptionCandidate(
                text=result.summary,
                source=DescriptionSource.AI_SUMMARY,
                confidence=result.confidence,
                quality_score=result.quality_score,
                length=len(result.summary),
                processing_time=result.processing_time,
                metadata={
                    'model_used': result.model_used,
                    'token_count': result.token_count,
                    'fallback_used': result.fallback_used,
                    'error_message': result.error_message
                }
            )
            
        except Exception as e:
            logger.debug(f"AI summary generation failed: {e}")
            return None
    
    def _extract_content_snippet(self, html_content: str, url: str, title: str) -> Optional[DescriptionCandidate]:
        """Extract content snippet candidate"""
        
        try:
            start_time = time.time()
            
            snippet = self.content_extractor.extract_content_snippet(html_content, url, title)
            
            if not snippet.text:
                return None
            
            processing_time = time.time() - start_time
            
            return DescriptionCandidate(
                text=snippet.text,
                source=DescriptionSource.CONTENT_SNIPPET,
                confidence=snippet.confidence,
                quality_score=snippet.quality_score,
                length=snippet.word_count,
                processing_time=processing_time,
                metadata={
                    'extraction_method': snippet.extraction_method,
                    'content_source': snippet.source
                }
            )
            
        except Exception as e:
            logger.debug(f"Content snippet extraction failed: {e}")
            return None
    
    def _create_fallback_description(self, title: str, url: str) -> DescriptionCandidate:
        """Create fallback description as last resort"""
        
        try:
            if title and len(title) > 10:
                description = f"Content titled '{title}'"
            elif url:
                from urllib.parse import urlparse
                domain = urlparse(url).netloc
                if domain:
                    description = f"Web content from {domain}"
                else:
                    description = "Web content"
            else:
                description = "Bookmark content"
            
            # Ensure proper ending
            if not description.endswith('.'):
                description += '.'
            
            return DescriptionCandidate(
                text=description,
                source=DescriptionSource.FALLBACK,
                confidence=0.2,
                quality_score=0.3,
                length=len(description),
                processing_time=0.0,
                metadata={'fallback_type': 'title_based' if title else 'generic'}
            )
            
        except Exception as e:
            logger.error(f"Fallback description creation failed: {e}")
            
            return DescriptionCandidate(
                text="Bookmark content.",
                source=DescriptionSource.FALLBACK,
                confidence=0.1,
                quality_score=0.2,
                length=17,
                processing_time=0.0,
                metadata={'fallback_type': 'emergency'}
            )
    
    def _select_best_description(self, candidates: List[DescriptionCandidate], 
                               existing_description: str) -> DescriptionCandidate:
        """Select the best description from candidates"""
        
        if not candidates:
            return self._create_fallback_description("", "")
        
        # Calculate composite scores for each candidate
        scored_candidates = []
        
        for candidate in candidates:
            # Base score from quality and confidence
            base_score = (candidate.quality_score * 0.6) + (candidate.confidence * 0.4)
            
            # Apply source priority
            priority_bonus = self.source_priorities.get(candidate.source, 0.5) * 0.2
            
            # Length appropriateness bonus
            length_bonus = 0.0
            if self.ideal_length_min <= candidate.length <= self.ideal_length_max:
                length_bonus = 0.1
            elif self.min_length <= candidate.length <= self.max_length:
                length_bonus = 0.05
            
            # User preservation bonus
            if candidate.source == DescriptionSource.USER_EXISTING and self.preserve_user_descriptions:
                preservation_bonus = 0.3
            else:
                preservation_bonus = 0.0
            
            # Calculate final score
            final_score = base_score + priority_bonus + length_bonus + preservation_bonus
            
            scored_candidates.append((candidate, final_score))
        
        # Sort by score and return best
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        best_candidate = scored_candidates[0][0]
        
        logger.debug(f"Selected {best_candidate.source.value} as best description source")
        return best_candidate
    
    def _apply_preservation_and_merging(self, 
                                      best_candidate: DescriptionCandidate,
                                      user_candidate: Optional[DescriptionCandidate],
                                      all_candidates: List[DescriptionCandidate]) -> Tuple[DescriptionCandidate, bool, bool]:
        """Apply preservation and merging logic"""
        
        preservation_applied = False
        merging_applied = False
        
        # Check if we should preserve user description
        if (user_candidate and 
            self.preserve_user_descriptions and 
            user_candidate.quality_score >= self.user_description_quality_threshold):
            
            # If user description is good enough, preserve it
            if (best_candidate.source != DescriptionSource.USER_EXISTING and
                user_candidate.quality_score >= best_candidate.quality_score - self.quality_improvement_threshold):
                
                logger.debug("Preserving user description due to quality threshold")
                preservation_applied = True
                return user_candidate, preservation_applied, merging_applied
        
        # Check if we should merge descriptions
        if (user_candidate and 
            best_candidate.source != DescriptionSource.USER_EXISTING and
            user_candidate.quality_score >= 0.4 and
            best_candidate.quality_score >= 0.6):
            
            # Try to merge user description with generated description
            merged_description = self._merge_descriptions(user_candidate.text, best_candidate.text)
            
            if merged_description and merged_description != best_candidate.text:
                logger.debug("Applied description merging")
                merging_applied = True
                
                # Create merged candidate
                merged_candidate = DescriptionCandidate(
                    text=merged_description,
                    source=best_candidate.source,
                    confidence=min(user_candidate.confidence, best_candidate.confidence) + 0.1,
                    quality_score=(user_candidate.quality_score + best_candidate.quality_score) / 2,
                    length=len(merged_description),
                    processing_time=best_candidate.processing_time,
                    metadata={
                        'merged_from': [user_candidate.source.value, best_candidate.source.value],
                        **best_candidate.metadata
                    }
                )
                
                return merged_candidate, preservation_applied, merging_applied
        
        return best_candidate, preservation_applied, merging_applied
    
    def _merge_descriptions(self, user_description: str, generated_description: str) -> Optional[str]:
        """Intelligently merge user and generated descriptions"""
        
        try:
            # Clean both descriptions
            user_clean = user_description.strip()
            generated_clean = generated_description.strip()
            
            # If user description is very short, append generated content
            if len(user_clean) < 50 and len(generated_clean) > 50:
                merged = f"{user_clean} {generated_clean}"
                
                # Ensure proper sentence structure
                if not user_clean.endswith('.'):
                    merged = f"{user_clean}. {generated_clean}"
                
                # Check length
                if len(merged) <= self.max_length:
                    return merged
            
            # If generated description is much more informative, prefer it but keep user context
            user_words = set(user_clean.lower().split())
            generated_words = set(generated_clean.lower().split())
            
            # Check for complementary information
            unique_user_words = user_words - generated_words
            unique_generated_words = generated_words - user_words
            
            # If they have different information, try to combine
            if len(unique_user_words) > 2 and len(unique_generated_words) > 5:
                # Take first sentence from user, rest from generated
                user_sentences = user_clean.split('.')
                generated_sentences = generated_clean.split('.')
                
                if user_sentences and generated_sentences:
                    first_user_sentence = user_sentences[0].strip()
                    
                    if len(first_user_sentence) > 10:
                        # Combine first user sentence with generated description
                        merged = f"{first_user_sentence}. {generated_clean}"
                        
                        if len(merged) <= self.max_length:
                            return merged
            
            # If no good merge possible, return None to use original best candidate
            return None
            
        except Exception as e:
            logger.debug(f"Description merging failed: {e}")
            return None
    
    def _validate_and_format_description(self, description: str) -> Tuple[str, bool, List[str]]:
        """Validate and format the final description"""
        
        issues = []
        
        try:
            # Clean the description
            formatted = description.strip()
            
            # Length validation
            if len(formatted) < self.min_length:
                issues.append(f"Description too short (minimum {self.min_length} characters)")
            
            if len(formatted) > self.max_length:
                # Truncate intelligently
                formatted = self._smart_truncate_description(formatted)
                issues.append("Description was truncated to fit length limit")
            
            # Ensure proper capitalization
            if formatted and not formatted[0].isupper():
                formatted = formatted[0].upper() + formatted[1:]
            
            # Ensure proper ending punctuation
            if formatted and not formatted.endswith(('.', '!', '?', '…')):
                formatted += '.'
            
            # Remove duplicate punctuation
            formatted = re.sub(r'[.!?]{2,}', '.', formatted)
            
            # Check for quality issues
            if len(formatted.split()) < 5:
                issues.append("Description has very few words")
            
            # Check for repetitive content
            words = formatted.lower().split()
            if len(words) > 5:
                word_counts = {}
                for word in words:
                    if len(word) > 3:  # Only check meaningful words
                        word_counts[word] = word_counts.get(word, 0) + 1
                
                repeated_words = [word for word, count in word_counts.items() if count > 2]
                if repeated_words:
                    issues.append(f"Repetitive words found: {', '.join(repeated_words[:3])}")
            
            validation_passed = len(issues) == 0
            
            return formatted, validation_passed, issues
            
        except Exception as e:
            logger.error(f"Description validation failed: {e}")
            issues.append(f"Validation failed: {e}")
            return description, False, issues
    
    def _smart_truncate_description(self, description: str) -> str:
        """Intelligently truncate description to fit length limit"""
        
        if len(description) <= self.max_length:
            return description
        
        # Try to truncate at sentence boundary
        sentences = description.split('.')
        truncated = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(truncated + sentence + '.') <= self.max_length - 3:
                if truncated:
                    truncated += '. ' + sentence
                else:
                    truncated = sentence
            else:
                break
        
        if truncated and len(truncated) >= self.min_length:
            return truncated + '.'
        else:
            # Hard truncate with ellipsis
            return description[:self.max_length-3] + "..."
    
    def _calculate_description_quality(self, description: str, source: str) -> float:
        """Calculate quality score for a description"""
        
        if not description:
            return 0.0
        
        score = 0.0
        
        # Length scoring (0.3 weight)
        length = len(description)
        if self.ideal_length_min <= length <= self.ideal_length_max:
            length_score = 1.0
        elif self.min_length <= length <= self.max_length:
            length_score = 0.8
        elif length >= self.min_length:
            length_score = 0.6
        else:
            length_score = length / self.min_length
        
        score += length_score * 0.3
        
        # Content quality (0.4 weight)
        content_score = 0.0
        
        # Check for complete sentences
        if description.endswith(('.', '!', '?')):
            content_score += 0.2
        
        # Check for proper capitalization
        if description and description[0].isupper():
            content_score += 0.1
        
        # Check for descriptive content
        descriptive_words = ['provides', 'offers', 'features', 'includes', 'helps', 'enables', 'allows', 'supports']
        if any(word in description.lower() for word in descriptive_words):
            content_score += 0.3
        
        # Check for specific information
        if re.search(r'\d+', description):
            content_score += 0.1
        
        # Check for proper nouns (likely to be informative)
        if re.search(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', description):
            content_score += 0.2
        
        # Penalty for generic phrases
        generic_phrases = ['welcome to', 'home page', 'click here', 'read more', 'learn more']
        for phrase in generic_phrases:
            if phrase in description.lower():
                content_score -= 0.2
        
        content_score = max(0.0, min(1.0, content_score))
        score += content_score * 0.4
        
        # Readability (0.2 weight)
        words = description.split()
        if words:
            avg_word_length = sum(len(word) for word in words) / len(words)
            # Prefer moderate word length
            if 4 <= avg_word_length <= 7:
                readability_score = 1.0
            else:
                readability_score = max(0.3, 1.0 - abs(avg_word_length - 5.5) / 5)
        else:
            readability_score = 0.0
        
        score += readability_score * 0.2
        
        # Source bonus (0.1 weight)
        source_bonus = {
            'user_existing': 0.2,
            'meta_description': 0.1,
            'ai_summary': 0.0,
            'content_snippet': -0.1,
            'fallback': -0.2
        }.get(source, 0.0)
        
        score += source_bonus * 0.1
        
        return max(0.0, min(1.0, score))
    
    def batch_generate_descriptions(self, 
                                  items: List[Dict[str, Any]]) -> List[DescriptionGenerationResult]:
        """Generate descriptions for multiple items"""
        
        results = []
        
        for item in items:
            try:
                result = self.generate_description(
                    title=item.get('title', ''),
                    content=item.get('content', ''),
                    url=item.get('url', ''),
                    existing_description=item.get('existing_description', ''),
                    html_content=item.get('html_content', '')
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Batch description generation failed for item: {e}")
                
                # Create error result
                error_result = DescriptionGenerationResult(
                    final_description="Description generation failed.",
                    source_used=DescriptionSource.FALLBACK,
                    confidence=0.0,
                    quality_score=0.0,
                    processing_time=0.0,
                    candidates_evaluated=[],
                    preservation_applied=False,
                    merging_applied=False,
                    validation_passed=False,
                    validation_issues=[f"Generation failed: {e}"]
                )
                results.append(error_result)
        
        return results
    
    def get_generator_stats(self) -> Dict[str, Any]:
        """Get description generator statistics and configuration"""
        
        return {
            'min_length': self.min_length,
            'max_length': self.max_length,
            'ideal_length_range': f"{self.ideal_length_min}-{self.ideal_length_max}",
            'preserve_user_descriptions': self.preserve_user_descriptions,
            'user_quality_threshold': self.user_description_quality_threshold,
            'min_confidence_threshold': self.min_confidence_threshold,
            'quality_improvement_threshold': self.quality_improvement_threshold,
            'enabled_sources': {
                'meta_extraction': self.enable_meta_extraction,
                'ai_summarization': self.enable_ai_summarization,
                'content_extraction': self.enable_content_extraction
            },
            'source_priorities': {source.value: priority for source, priority in self.source_priorities.items()},
            'component_stats': {
                'meta_extractor': self.meta_extractor.get_extraction_stats(),
                'ai_summarizer': self.ai_summarizer.get_summarizer_stats(),
                'content_extractor': self.content_extractor.get_extraction_stats()
            }
        }
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            if self.ai_summarizer:
                self.ai_summarizer.cleanup()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")


# Import re module at the top
import re