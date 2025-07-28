"""AI-powered content summarization system using Ollama"""

import re
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urlparse

from ai.ollama_client import OllamaClient, OllamaResponse
from utils.logging_utils import get_logger
from utils.text_utils import TextUtils

logger = get_logger(__name__)


@dataclass
class SummarizationResult:
    """Result from AI summarization"""
    summary: Optional[str] = None
    model_used: str = ""
    processing_time: float = 0.0
    token_count: int = 0
    confidence: float = 0.0
    quality_score: float = 0.0
    fallback_used: bool = False
    error_message: Optional[str] = None


@dataclass
class ContentPreprocessingResult:
    """Result from content preprocessing"""
    cleaned_content: str
    extracted_sections: Dict[str, str]
    content_type: str
    language: str
    word_count: int
    key_phrases: List[str]


class AISummarizer:
    """AI-powered content summarization using local LLM via Ollama"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize AI summarizer"""
        self.config = config
        self.ai_config = config.get('ai', {})
        self.ollama_config = self.ai_config.get('ollama', {})
        
        # Initialize Ollama client
        self.ollama_client = OllamaClient(
            host=self.ollama_config.get('host', 'localhost:11434'),
            model_name=self.ollama_config.get('model', 'llama2'),
            auto_start=self.ollama_config.get('auto_start', True),
            auto_pull=self.ollama_config.get('auto_pull', True)
        )
        
        # Summarization settings
        self.default_max_length = self.ai_config.get('summary_max_length', 200)
        self.min_content_length = self.ai_config.get('min_content_for_summary', 100)
        self.max_content_length = self.ai_config.get('max_content_for_summary', 5000)
        self.temperature = self.ai_config.get('temperature', 0.3)
        self.max_tokens = self.ai_config.get('max_summary_tokens', 100)
        
        # Quality thresholds
        self.min_quality_score = 0.6
        self.min_confidence_score = 0.7
        
        # Content preprocessing patterns
        self.content_patterns = {
            'code_blocks': r'```[\s\S]*?```',
            'inline_code': r'`[^`]+`',
            'urls': r'https?://[^\s]+',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'html_tags': r'<[^>]+>',
            'markdown_headers': r'^#{1,6}\s+.+$',
            'list_items': r'^\s*[-*+]\s+.+$',
            'numbered_lists': r'^\s*\d+\.\s+.+$'
        }
        
        # Domain-specific prompts
        self.domain_prompts = {
            'github.com': self._get_github_prompt,
            'stackoverflow.com': self._get_stackoverflow_prompt,
            'medium.com': self._get_article_prompt,
            'dev.to': self._get_article_prompt,
            'news': self._get_news_prompt,
            'blog': self._get_blog_prompt,
            'documentation': self._get_docs_prompt,
            'tutorial': self._get_tutorial_prompt,
            'default': self._get_default_prompt
        }
        
        logger.info("AI summarizer initialized with Ollama client")
    
    def summarize_content(self, 
                         title: str, 
                         content: str, 
                         url: str = "",
                         max_length: int = None,
                         content_type: str = "general") -> SummarizationResult:
        """Generate AI summary for content"""
        
        start_time = time.time()
        max_length = max_length or self.default_max_length
        
        try:
            # Validate inputs
            if not content or len(content.strip()) < self.min_content_length:
                return SummarizationResult(
                    error_message=f"Content too short for summarization (minimum {self.min_content_length} characters)"
                )
            
            # Preprocess content
            preprocessing_result = self._preprocess_content(title, content, url)
            
            if not preprocessing_result.cleaned_content:
                return SummarizationResult(
                    error_message="Content preprocessing failed"
                )
            
            # Generate domain-specific prompt
            prompt = self._generate_prompt(
                title, 
                preprocessing_result.cleaned_content, 
                url, 
                max_length, 
                content_type,
                preprocessing_result
            )
            
            # Generate summary using Ollama
            response = self.ollama_client.generate_response(
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            if not response or not response.content:
                return SummarizationResult(
                    error_message="Failed to generate summary from AI model"
                )
            
            # Post-process and validate summary
            processed_summary = self._post_process_summary(
                response.content, 
                max_length, 
                title
            )
            
            if not processed_summary:
                return SummarizationResult(
                    error_message="Summary post-processing failed"
                )
            
            # Calculate quality metrics
            quality_score = self._calculate_quality_score(
                processed_summary, 
                title, 
                preprocessing_result.cleaned_content
            )
            
            confidence = self._calculate_confidence(
                processed_summary, 
                response, 
                quality_score
            )
            
            processing_time = time.time() - start_time
            
            return SummarizationResult(
                summary=processed_summary,
                model_used=response.model,
                processing_time=processing_time,
                token_count=response.eval_count or 0,
                confidence=confidence,
                quality_score=quality_score,
                fallback_used=False
            )
            
        except Exception as e:
            logger.error(f"AI summarization failed: {e}")
            
            # Try fallback summarization
            fallback_result = self._fallback_summarization(title, content, max_length)
            fallback_result.processing_time = time.time() - start_time
            fallback_result.error_message = f"AI summarization failed: {e}"
            
            return fallback_result
    
    def _preprocess_content(self, title: str, content: str, url: str) -> ContentPreprocessingResult:
        """Preprocess content for better summarization"""
        
        try:
            # Clean basic text
            cleaned = TextUtils.clean_text(content)
            
            # Extract different content sections
            sections = {}
            
            # Extract and temporarily remove code blocks
            code_blocks = re.findall(self.content_patterns['code_blocks'], cleaned, re.MULTILINE)
            if code_blocks:
                sections['code_blocks'] = code_blocks
                cleaned = re.sub(self.content_patterns['code_blocks'], '[CODE_BLOCK]', cleaned, flags=re.MULTILINE)
            
            # Extract URLs
            urls = re.findall(self.content_patterns['urls'], cleaned)
            if urls:
                sections['urls'] = urls[:5]  # Keep only first 5 URLs
                cleaned = re.sub(self.content_patterns['urls'], '[URL]', cleaned)
            
            # Remove HTML tags
            cleaned = re.sub(self.content_patterns['html_tags'], ' ', cleaned)
            
            # Extract markdown headers
            headers = re.findall(self.content_patterns['markdown_headers'], cleaned, re.MULTILINE)
            if headers:
                sections['headers'] = headers
            
            # Extract list items
            list_items = re.findall(self.content_patterns['list_items'], cleaned, re.MULTILINE)
            numbered_items = re.findall(self.content_patterns['numbered_lists'], cleaned, re.MULTILINE)
            if list_items or numbered_items:
                sections['lists'] = list_items + numbered_items
            
            # Normalize whitespace
            cleaned = ' '.join(cleaned.split())
            
            # Truncate if too long
            if len(cleaned) > self.max_content_length:
                cleaned = cleaned[:self.max_content_length] + "..."
            
            # Detect content type
            content_type = self._detect_content_type(title, cleaned, url)
            
            # Detect language
            language = self._detect_language(cleaned)
            
            # Extract key phrases
            key_phrases = TextUtils.extract_keywords(cleaned, max_keywords=10)
            
            return ContentPreprocessingResult(
                cleaned_content=cleaned,
                extracted_sections=sections,
                content_type=content_type,
                language=language,
                word_count=len(cleaned.split()),
                key_phrases=key_phrases
            )
            
        except Exception as e:
            logger.error(f"Content preprocessing failed: {e}")
            return ContentPreprocessingResult(
                cleaned_content=content[:self.max_content_length],
                extracted_sections={},
                content_type="general",
                language="en",
                word_count=len(content.split()),
                key_phrases=[]
            )
    
    def _generate_prompt(self, 
                        title: str, 
                        content: str, 
                        url: str, 
                        max_length: int,
                        content_type: str,
                        preprocessing_result: ContentPreprocessingResult) -> str:
        """Generate domain-specific prompt for summarization"""
        
        # Determine domain
        domain = ""
        if url:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
        
        # Get appropriate prompt generator
        prompt_generator = None
        for domain_key, generator in self.domain_prompts.items():
            if domain_key in domain or domain_key == content_type:
                prompt_generator = generator
                break
        
        if not prompt_generator:
            prompt_generator = self.domain_prompts['default']
        
        return prompt_generator(title, content, url, max_length, preprocessing_result)
    
    def _get_github_prompt(self, title: str, content: str, url: str, max_length: int, 
                          preprocessing: ContentPreprocessingResult) -> str:
        """Generate prompt for GitHub repositories"""
        
        return f"""You are an expert at summarizing software repositories and development projects.

Create a concise summary of this GitHub repository in exactly {max_length} characters or less.
Focus on: what the project does, key technologies used, main features, and target audience.

Repository: {title}
URL: {url}
Content: {content[:1500]}

Key technologies found: {', '.join(preprocessing.key_phrases[:5])}

Write a clear, informative summary that helps developers understand what this repository offers:"""
    
    def _get_stackoverflow_prompt(self, title: str, content: str, url: str, max_length: int,
                                 preprocessing: ContentPreprocessingResult) -> str:
        """Generate prompt for Stack Overflow questions/answers"""
        
        return f"""You are an expert at summarizing programming questions and solutions.

Create a concise summary of this Stack Overflow content in exactly {max_length} characters or less.
Focus on: the programming problem, key technologies involved, and the solution approach.

Question/Answer: {title}
Content: {content[:1500]}

Technologies: {', '.join(preprocessing.key_phrases[:5])}

Write a clear summary that explains the programming problem and solution:"""
    
    def _get_article_prompt(self, title: str, content: str, url: str, max_length: int,
                           preprocessing: ContentPreprocessingResult) -> str:
        """Generate prompt for articles and blog posts"""
        
        return f"""You are an expert at summarizing articles and blog posts.

Create a concise summary of this article in exactly {max_length} characters or less.
Focus on: main topic, key insights, practical takeaways, and target audience.

Article: {title}
Content: {content[:1500]}

Key topics: {', '.join(preprocessing.key_phrases[:5])}

Write an engaging summary that captures the article's main value and insights:"""
    
    def _get_news_prompt(self, title: str, content: str, url: str, max_length: int,
                        preprocessing: ContentPreprocessingResult) -> str:
        """Generate prompt for news articles"""
        
        return f"""You are an expert at summarizing news articles.

Create a concise summary of this news article in exactly {max_length} characters or less.
Focus on: who, what, when, where, why, and the significance of the story.

Headline: {title}
Content: {content[:1500]}

Write a clear, factual summary that captures the essential news information:"""
    
    def _get_blog_prompt(self, title: str, content: str, url: str, max_length: int,
                        preprocessing: ContentPreprocessingResult) -> str:
        """Generate prompt for blog posts"""
        
        return f"""You are an expert at summarizing blog posts and personal content.

Create a concise summary of this blog post in exactly {max_length} characters or less.
Focus on: main message, personal insights, practical advice, and key takeaways.

Blog Post: {title}
Content: {content[:1500]}

Key themes: {', '.join(preprocessing.key_phrases[:5])}

Write an engaging summary that captures the blogger's main message and value:"""
    
    def _get_docs_prompt(self, title: str, content: str, url: str, max_length: int,
                        preprocessing: ContentPreprocessingResult) -> str:
        """Generate prompt for documentation"""
        
        return f"""You are an expert at summarizing technical documentation.

Create a concise summary of this documentation in exactly {max_length} characters or less.
Focus on: what the tool/API does, key features, usage scenarios, and target users.

Documentation: {title}
Content: {content[:1500]}

Technologies: {', '.join(preprocessing.key_phrases[:5])}

Write a clear, technical summary that explains the purpose and capabilities:"""
    
    def _get_tutorial_prompt(self, title: str, content: str, url: str, max_length: int,
                            preprocessing: ContentPreprocessingResult) -> str:
        """Generate prompt for tutorials"""
        
        return f"""You are an expert at summarizing tutorials and educational content.

Create a concise summary of this tutorial in exactly {max_length} characters or less.
Focus on: what you'll learn, key skills covered, prerequisites, and learning outcomes.

Tutorial: {title}
Content: {content[:1500]}

Skills/Topics: {', '.join(preprocessing.key_phrases[:5])}

Write an educational summary that explains what learners will gain from this tutorial:"""
    
    def _get_default_prompt(self, title: str, content: str, url: str, max_length: int,
                           preprocessing: ContentPreprocessingResult) -> str:
        """Generate default prompt for general content"""
        
        return f"""You are an expert at creating concise, informative summaries of web content.

Create a summary of this content in exactly {max_length} characters or less.
Focus on: main purpose, key information, value proposition, and target audience.

Title: {title}
URL: {url}
Content: {content[:1500]}

Key topics: {', '.join(preprocessing.key_phrases[:5])}

Write a clear, informative summary that captures the essential value and purpose:"""
    
    def _post_process_summary(self, raw_summary: str, max_length: int, title: str) -> str:
        """Post-process and clean the generated summary"""
        
        if not raw_summary:
            return ""
        
        # Clean the summary
        summary = raw_summary.strip()
        
        # Remove common AI response prefixes
        prefixes_to_remove = [
            'summary:', 'here is a summary:', 'this content is about:',
            'the summary is:', 'in summary:', 'to summarize:',
            'here\'s a summary:', 'summary of the content:'
        ]
        
        for prefix in prefixes_to_remove:
            if summary.lower().startswith(prefix):
                summary = summary[len(prefix):].strip()
        
        # Remove quotes if the entire summary is quoted
        if summary.startswith('"') and summary.endswith('"'):
            summary = summary[1:-1].strip()
        
        # Ensure proper sentence structure
        if summary and not summary[0].isupper():
            summary = summary[0].upper() + summary[1:]
        
        # Ensure proper ending punctuation
        if summary and not summary.endswith(('.', '!', '?', '…')):
            summary += '.'
        
        # Truncate if too long
        if len(summary) > max_length:
            # Try to truncate at sentence boundary
            sentences = summary.split('.')
            truncated = ""
            for sentence in sentences:
                if len(truncated + sentence + '.') <= max_length - 3:
                    truncated += sentence + '.'
                else:
                    break
            
            if truncated:
                summary = truncated
            else:
                # Hard truncate with ellipsis
                summary = summary[:max_length-3] + "..."
        
        # Final validation
        if len(summary) < 20:  # Too short to be useful
            return ""
        
        return summary
    
    def _calculate_quality_score(self, summary: str, title: str, original_content: str) -> float:
        """Calculate quality score for generated summary"""
        
        if not summary:
            return 0.0
        
        score = 0.0
        
        # Length appropriateness (0.2 weight)
        length = len(summary)
        if 80 <= length <= 200:
            length_score = 1.0
        elif length < 80:
            length_score = length / 80
        else:
            length_score = max(0.3, 1.0 - (length - 200) / 100)
        
        score += length_score * 0.2
        
        # Content coverage (0.3 weight)
        # Check if summary contains key concepts from original
        original_words = set(original_content.lower().split())
        summary_words = set(summary.lower().split())
        
        # Remove common stop words for better analysis
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
        
        original_content_words = original_words - stop_words
        summary_content_words = summary_words - stop_words
        
        if original_content_words:
            coverage = len(summary_content_words & original_content_words) / len(original_content_words)
            coverage_score = min(1.0, coverage * 3)  # Scale up since summaries are much shorter
        else:
            coverage_score = 0.5
        
        score += coverage_score * 0.3
        
        # Coherence and readability (0.3 weight)
        coherence_score = 0.0
        
        # Check for complete sentences
        sentences = [s.strip() for s in summary.split('.') if s.strip()]
        if sentences:
            complete_sentences = sum(1 for s in sentences if len(s) > 10 and ' ' in s)
            coherence_score += (complete_sentences / len(sentences)) * 0.5
        
        # Check for proper capitalization
        if summary and summary[0].isupper():
            coherence_score += 0.2
        
        # Check for proper punctuation
        if summary and summary.endswith(('.', '!', '?')):
            coherence_score += 0.2
        
        # Penalty for repetitive content
        words = summary.lower().split()
        if len(words) > 5:
            unique_words = len(set(words))
            repetition_penalty = max(0, 1 - (len(words) - unique_words) / len(words))
            coherence_score *= repetition_penalty
        
        coherence_score = min(1.0, coherence_score)
        score += coherence_score * 0.3
        
        # Informativeness (0.2 weight)
        info_score = 0.0
        
        # Check for specific information (numbers, proper nouns, technical terms)
        if re.search(r'\d+', summary):
            info_score += 0.3
        
        # Check for descriptive verbs
        descriptive_verbs = ['provides', 'offers', 'enables', 'helps', 'supports', 'includes', 'features', 'allows', 'delivers', 'creates', 'builds', 'implements']
        if any(verb in summary.lower() for verb in descriptive_verbs):
            info_score += 0.3
        
        # Check for technical terms or domain-specific language
        if len([word for word in summary.split() if len(word) > 6]) > 2:
            info_score += 0.2
        
        # Check if it's not just a rephrasing of the title
        title_words = set(title.lower().split())
        summary_words = set(summary.lower().split())
        if title_words and len(summary_words - title_words) > len(title_words):
            info_score += 0.2
        
        info_score = min(1.0, info_score)
        score += info_score * 0.2
        
        return max(0.0, min(1.0, score))
    
    def _calculate_confidence(self, summary: str, response: OllamaResponse, quality_score: float) -> float:
        """Calculate confidence in the generated summary"""
        
        confidence = 0.0
        
        # Base confidence from model response
        if response.done:
            confidence += 0.3
        
        # Token count indicates model engagement
        if response.eval_count and response.eval_count > 20:
            confidence += 0.2
        
        # Quality score contribution
        confidence += quality_score * 0.4
        
        # Length appropriateness
        if 50 <= len(summary) <= 250:
            confidence += 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def _detect_content_type(self, title: str, content: str, url: str) -> str:
        """Detect the type of content for better summarization"""
        
        text = f"{title} {content} {url}".lower()
        
        # Content type patterns
        patterns = {
            'github': ['github.com', 'repository', 'repo', 'source code', 'git'],
            'stackoverflow': ['stackoverflow.com', 'stack overflow', 'programming question'],
            'documentation': ['docs', 'documentation', 'api', 'reference', 'manual', 'guide'],
            'tutorial': ['tutorial', 'how to', 'step by step', 'learn', 'course', 'lesson'],
            'news': ['news', 'breaking', 'report', 'journalist', 'press'],
            'blog': ['blog', 'post', 'personal', 'thoughts', 'opinion'],
            'article': ['article', 'medium.com', 'dev.to', 'essay', 'analysis']
        }
        
        # Score each content type
        scores = {}
        for content_type, keywords in patterns.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                scores[content_type] = score
        
        # Return highest scoring type or 'general'
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        else:
            return 'general'
    
    def _detect_language(self, content: str) -> str:
        """Simple language detection"""
        
        # Very basic language detection based on common words
        english_indicators = ['the', 'and', 'or', 'but', 'with', 'for', 'this', 'that', 'have', 'will']
        
        content_lower = content.lower()
        english_score = sum(1 for word in english_indicators if word in content_lower)
        
        # Default to English for now (could be enhanced with proper language detection)
        return 'en' if english_score > 2 else 'unknown'
    
    def _fallback_summarization(self, title: str, content: str, max_length: int) -> SummarizationResult:
        """Fallback summarization when AI fails"""
        
        try:
            # Simple extractive summarization
            sentences = content.split('.')
            
            # Score sentences by position and content
            scored_sentences = []
            for i, sentence in enumerate(sentences[:10]):  # Only consider first 10 sentences
                sentence = sentence.strip()
                if len(sentence) < 20:  # Skip very short sentences
                    continue
                
                score = 0.0
                
                # Position score (earlier sentences are more important)
                score += (10 - i) / 10 * 0.3
                
                # Length score (prefer moderate length)
                if 50 <= len(sentence) <= 150:
                    score += 0.3
                
                # Title word overlap
                title_words = set(title.lower().split())
                sentence_words = set(sentence.lower().split())
                if title_words and sentence_words:
                    overlap = len(title_words & sentence_words) / len(title_words)
                    score += overlap * 0.4
                
                scored_sentences.append((sentence, score))
            
            if not scored_sentences:
                # Ultimate fallback
                summary = f"Summary of {title}"
                if len(summary) > max_length:
                    summary = summary[:max_length-3] + "..."
                
                return SummarizationResult(
                    summary=summary,
                    model_used="fallback",
                    confidence=0.2,
                    quality_score=0.3,
                    fallback_used=True
                )
            
            # Select best sentences
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            
            summary = ""
            for sentence, _ in scored_sentences:
                if len(summary + sentence + '. ') <= max_length:
                    summary += sentence + '. '
                else:
                    break
            
            summary = summary.strip()
            if not summary.endswith('.'):
                summary += '.'
            
            return SummarizationResult(
                summary=summary,
                model_used="fallback_extractive",
                confidence=0.4,
                quality_score=0.5,
                fallback_used=True
            )
            
        except Exception as e:
            logger.error(f"Fallback summarization failed: {e}")
            
            # Ultimate fallback
            summary = f"Summary of {title}"
            if len(summary) > max_length:
                summary = summary[:max_length-3] + "..."
            
            return SummarizationResult(
                summary=summary,
                model_used="fallback_title",
                confidence=0.1,
                quality_score=0.2,
                fallback_used=True
            )
    
    def batch_summarize(self, 
                       items: List[Tuple[str, str, str]], 
                       max_length: int = None) -> List[SummarizationResult]:
        """Batch summarize multiple items"""
        
        results = []
        max_length = max_length or self.default_max_length
        
        for title, content, url in items:
            try:
                result = self.summarize_content(title, content, url, max_length)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch summarization failed for {title}: {e}")
                results.append(SummarizationResult(
                    error_message=f"Batch summarization failed: {e}",
                    fallback_used=True
                ))
        
        return results
    
    def get_summarizer_stats(self) -> Dict[str, Any]:
        """Get summarizer statistics and configuration"""
        
        ollama_info = self.ollama_client.get_server_info()
        
        return {
            'ollama_status': ollama_info.get('status', 'unknown'),
            'available_models': ollama_info.get('models', []),
            'default_model': self.ollama_client.model_name,
            'default_max_length': self.default_max_length,
            'min_content_length': self.min_content_length,
            'max_content_length': self.max_content_length,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'supported_domains': list(self.domain_prompts.keys()),
            'min_quality_score': self.min_quality_score,
            'min_confidence_score': self.min_confidence_score
        }
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        if self.ollama_client:
            self.ollama_client.cleanup()