"""Interactive CLI components for user feedback and review"""

import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from intelligence.adaptive_intelligence import AdaptiveIntelligence, FeedbackType
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class InteractiveReviewer:
    """Interactive reviewer for suggestions and feedback"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize interactive reviewer"""
        self.config = config
        self.adaptive_intelligence = AdaptiveIntelligence(
            config.get('directories', {}).get('data_dir', 'data')
        )
        
        # Review settings
        self.auto_accept_threshold = config.get('interactive', {}).get('auto_accept_threshold', 0.9)
        self.show_confidence_scores = config.get('interactive', {}).get('show_confidence', True)
        self.max_suggestions_to_show = config.get('interactive', {}).get('max_suggestions', 10)
        
        logger.info("Interactive reviewer initialized")
    
    def review_category_suggestions(self, url: str, title: str, content: str,
                                  suggestions: List[Tuple[str, float]]) -> Optional[str]:
        """Interactively review category suggestions"""
        
        try:
            if not suggestions:
                print("No category suggestions available.")
                return None
            
            print(f"\n📂 Category Suggestions for: {title}")
            print(f"🔗 URL: {url}")
            print("-" * 60)
            
            # Show suggestions
            for i, (category, confidence) in enumerate(suggestions[:self.max_suggestions_to_show], 1):
                confidence_display = f"({confidence:.1%})" if self.show_confidence_scores else ""
                print(f"{i}. {category} {confidence_display}")
            
            print(f"{len(suggestions) + 1}. Enter custom category")
            print(f"{len(suggestions) + 2}. Skip (no category)")
            
            # Get user choice
            while True:
                try:
                    choice = input(f"\nSelect category (1-{len(suggestions) + 2}): ").strip()
                    
                    if not choice:
                        continue
                    
                    choice_num = int(choice)
                    
                    if 1 <= choice_num <= len(suggestions):
                        # User selected a suggestion
                        selected_category, confidence = suggestions[choice_num - 1]
                        
                        # Track feedback
                        self._track_category_feedback(
                            url, title, content, suggestions, selected_category, 
                            FeedbackType.SUGGESTION_ACCEPTED, confidence
                        )
                        
                        print(f"✅ Selected: {selected_category}")
                        return selected_category
                    
                    elif choice_num == len(suggestions) + 1:
                        # Custom category
                        custom_category = input("Enter custom category: ").strip()
                        if custom_category:
                            # Track as modification
                            original_suggestion = suggestions[0][0] if suggestions else ""
                            self._track_category_feedback(
                                url, title, content, suggestions, custom_category,
                                FeedbackType.SUGGESTION_MODIFIED, suggestions[0][1] if suggestions else 0.5
                            )
                            
                            print(f"✅ Custom category: {custom_category}")
                            return custom_category
                    
                    elif choice_num == len(suggestions) + 2:
                        # Skip
                        if suggestions:
                            self._track_category_feedback(
                                url, title, content, suggestions, "",
                                FeedbackType.SUGGESTION_REJECTED, suggestions[0][1]
                            )
                        print("⏭️ Skipped category selection")
                        return None
                    
                    else:
                        print(f"Invalid choice. Please enter 1-{len(suggestions) + 2}")
                
                except ValueError:
                    print("Please enter a valid number")
                except KeyboardInterrupt:
                    print("\n❌ Cancelled")
                    return None
            
        except Exception as e:
            logger.error(f"Failed to review category suggestions: {e}")
            return None
    
    def review_tag_suggestions(self, url: str, title: str, content: str,
                             suggestions: List[Tuple[str, float]], 
                             existing_tags: List[str] = None) -> List[str]:
        """Interactively review tag suggestions"""
        
        try:
            if existing_tags is None:
                existing_tags = []
            
            if not suggestions:
                print("No tag suggestions available.")
                return existing_tags
            
            print(f"\n🏷️ Tag Suggestions for: {title}")
            print(f"🔗 URL: {url}")
            if existing_tags:
                print(f"📌 Existing tags: {', '.join(existing_tags)}")
            print("-" * 60)
            
            selected_tags = existing_tags.copy()
            
            # Show suggestions with selection interface
            for i, (tag, confidence) in enumerate(suggestions[:self.max_suggestions_to_show], 1):
                if tag in existing_tags:
                    continue  # Skip existing tags
                
                confidence_display = f"({confidence:.1%})" if self.show_confidence_scores else ""
                print(f"{i}. {tag} {confidence_display}")
            
            print("\nOptions:")
            print("- Enter numbers separated by commas to select tags (e.g., 1,3,5)")
            print("- Type 'custom' to add custom tags")
            print("- Press Enter to finish")
            
            while True:
                try:
                    choice = input("\nYour selection: ").strip()
                    
                    if not choice:
                        # Finished selecting
                        break
                    
                    if choice.lower() == 'custom':
                        # Add custom tags
                        custom_input = input("Enter custom tags (comma-separated): ").strip()
                        if custom_input:
                            custom_tags = [tag.strip() for tag in custom_input.split(',') if tag.strip()]
                            for tag in custom_tags:
                                if tag not in selected_tags:
                                    selected_tags.append(tag)
                                    print(f"✅ Added custom tag: {tag}")
                        continue
                    
                    # Parse number selections
                    try:
                        numbers = [int(n.strip()) for n in choice.split(',')]
                        
                        for num in numbers:
                            if 1 <= num <= len(suggestions):
                                tag, confidence = suggestions[num - 1]
                                if tag not in selected_tags:
                                    selected_tags.append(tag)
                                    
                                    # Track feedback
                                    self._track_tag_feedback(
                                        url, title, content, [tag], [tag],
                                        FeedbackType.SUGGESTION_ACCEPTED, confidence
                                    )
                                    
                                    print(f"✅ Added: {tag}")
                                else:
                                    print(f"⚠️ Tag already exists: {tag}")
                            else:
                                print(f"❌ Invalid number: {num}")
                    
                    except ValueError:
                        print("❌ Invalid format. Use numbers separated by commas (e.g., 1,3,5)")
                
                except KeyboardInterrupt:
                    print("\n❌ Cancelled")
                    break
            
            # Track rejected suggestions
            accepted_tags = set(selected_tags) - set(existing_tags)
            rejected_suggestions = [tag for tag, _ in suggestions if tag not in accepted_tags]
            
            if rejected_suggestions:
                self._track_tag_feedback(
                    url, title, content, rejected_suggestions, [],
                    FeedbackType.SUGGESTION_REJECTED, 0.5
                )
            
            print(f"🏷️ Final tags: {', '.join(selected_tags)}")
            return selected_tags
            
        except Exception as e:
            logger.error(f"Failed to review tag suggestions: {e}")
            return existing_tags or []
    
    def review_enhancement_results(self, url: str, original_data: Dict[str, Any],
                                 enhanced_data: Dict[str, Any]) -> Dict[str, Any]:
        """Interactively review enhancement results"""
        
        try:
            print(f"\n🔍 Enhancement Results for: {url}")
            print("-" * 60)
            
            # Show original vs enhanced data
            print("📋 Original Data:")
            print(f"  Title: {original_data.get('name', 'N/A')}")
            print(f"  Description: {original_data.get('description', 'N/A')[:100]}...")
            
            print("\n✨ Enhanced Data:")
            print(f"  Title: {enhanced_data.get('name', 'N/A')}")
            print(f"  Description: {enhanced_data.get('description', 'N/A')[:100]}...")
            
            # Show enhancement metadata
            if 'enhancement_metadata' in enhanced_data:
                metadata = enhanced_data['enhancement_metadata']
                print(f"\n📊 Enhancement Info:")
                print(f"  Scraper used: {metadata.get('scraper_used', 'N/A')}")
                print(f"  Success: {metadata.get('success', False)}")
                print(f"  Processing time: {metadata.get('processing_time', 0):.2f}s")
            
            # Get user approval
            while True:
                try:
                    choice = input("\nAccept enhancement? (y/n/e for edit): ").strip().lower()
                    
                    if choice in ['y', 'yes']:
                        print("✅ Enhancement accepted")
                        return enhanced_data
                    
                    elif choice in ['n', 'no']:
                        print("❌ Enhancement rejected, keeping original")
                        return original_data
                    
                    elif choice in ['e', 'edit']:
                        return self._edit_enhancement_data(enhanced_data)
                    
                    else:
                        print("Please enter 'y' (yes), 'n' (no), or 'e' (edit)")
                
                except KeyboardInterrupt:
                    print("\n❌ Cancelled, keeping original")
                    return original_data
            
        except Exception as e:
            logger.error(f"Failed to review enhancement results: {e}")
            return original_data
    
    def _edit_enhancement_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Allow user to edit enhancement data"""
        
        try:
            edited_data = data.copy()
            
            print("\n✏️ Edit Enhancement Data:")
            print("Press Enter to keep current value, or type new value")
            
            # Edit title
            current_title = edited_data.get('name', '')
            new_title = input(f"Title [{current_title}]: ").strip()
            if new_title:
                edited_data['name'] = new_title
            
            # Edit description
            current_desc = edited_data.get('description', '')
            print(f"\nCurrent description: {current_desc}")
            new_desc = input("New description (or Enter to keep): ").strip()
            if new_desc:
                edited_data['description'] = new_desc
            
            print("✅ Enhancement data updated")
            return edited_data
            
        except Exception as e:
            logger.error(f"Failed to edit enhancement data: {e}")
            return data
    
    def _track_category_feedback(self, url: str, title: str, content: str,
                               suggestions: List[Tuple[str, float]], 
                               user_choice: str, feedback_type: FeedbackType,
                               confidence: float) -> None:
        """Track category feedback for learning"""
        
        try:
            context = {
                'url': url,
                'title': title,
                'content': content[:500],  # Limit content length
                'suggestion_type': 'category',
                'suggestions_count': len(suggestions)
            }
            
            original_suggestion = suggestions[0][0] if suggestions else ""
            
            self.adaptive_intelligence.track_user_feedback(
                feedback_type=feedback_type,
                context=context,
                original_suggestion=original_suggestion,
                user_action=user_choice,
                confidence_before=confidence
            )
            
        except Exception as e:
            logger.warning(f"Failed to track category feedback: {e}")
    
    def _track_tag_feedback(self, url: str, title: str, content: str,
                          suggested_tags: List[str], selected_tags: List[str],
                          feedback_type: FeedbackType, confidence: float) -> None:
        """Track tag feedback for learning"""
        
        try:
            context = {
                'url': url,
                'title': title,
                'content': content[:500],  # Limit content length
                'suggestion_type': 'tags',
                'suggestions_count': len(suggested_tags)
            }
            
            self.adaptive_intelligence.track_user_feedback(
                feedback_type=feedback_type,
                context=context,
                original_suggestion=suggested_tags,
                user_action=selected_tags,
                confidence_before=confidence
            )
            
        except Exception as e:
            logger.warning(f"Failed to track tag feedback: {e}")
    
    def show_learning_progress(self) -> None:
        """Show learning progress and statistics"""
        
        try:
            print("\n🧠 Learning Progress")
            print("-" * 40)
            
            stats = self.adaptive_intelligence.get_adaptation_statistics()
            
            print(f"Total preferences learned: {stats.get('total_preferences', 0)}")
            print(f"Strong preferences: {stats.get('strong_preferences', 0)}")
            print(f"Total feedback items: {stats.get('total_feedback_items', 0)}")
            print(f"Recent feedback (7 days): {stats.get('recent_feedback_count', 0)}")
            
            # Show suggestion performance
            performance = stats.get('suggestion_performance', {})
            if performance:
                print("\n📊 Suggestion Performance:")
                for suggestion_type, perf_data in performance.items():
                    acceptance_rate = perf_data.get('acceptance_rate', 0) * 100
                    total_interactions = perf_data.get('total_interactions', 0)
                    print(f"  {suggestion_type}: {acceptance_rate:.1f}% acceptance ({total_interactions} interactions)")
            
            # Show preference breakdown
            preference_breakdown = stats.get('preference_breakdown', {})
            if preference_breakdown:
                print("\n🎯 Preference Types:")
                for pref_type, count in preference_breakdown.items():
                    print(f"  {pref_type}: {count} preferences")
            
        except Exception as e:
            logger.error(f"Failed to show learning progress: {e}")
            print("❌ Unable to display learning progress")
    
    def get_user_confirmation(self, message: str, default: bool = True) -> bool:
        """Get yes/no confirmation from user"""
        
        try:
            default_text = "Y/n" if default else "y/N"
            
            while True:
                response = input(f"{message} ({default_text}): ").strip().lower()
                
                if not response:
                    return default
                
                if response in ['y', 'yes']:
                    return True
                elif response in ['n', 'no']:
                    return False
                else:
                    print("Please enter 'y' for yes or 'n' for no")
        
        except KeyboardInterrupt:
            print("\n❌ Cancelled")
            return False
    
    def show_progress_with_interaction(self, current: int, total: int, 
                                     current_item: str = "") -> None:
        """Show progress with option for user interaction"""
        
        try:
            percentage = (current / total) * 100 if total > 0 else 0
            
            # Create progress bar
            bar_length = 40
            filled_length = int(bar_length * current // total) if total > 0 else 0
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            
            # Show progress
            print(f"\r🔄 Progress: |{bar}| {percentage:.1f}% ({current}/{total}) {current_item}", end='', flush=True)
            
            # Check for user interrupt every 10 items
            if current % 10 == 0:
                # This allows for Ctrl+C to be caught
                pass
            
        except KeyboardInterrupt:
            print(f"\n⏸️ Process interrupted at {current}/{total}")
            raise
        except Exception as e:
            logger.warning(f"Failed to show progress: {e}")


class InteractiveMenu:
    """Interactive menu system for CLI"""
    
    def __init__(self):
        """Initialize interactive menu"""
        self.menu_stack = []
    
    def show_main_menu(self) -> str:
        """Show main menu and get user choice"""
        
        print("\n" + "="*60)
        print("🔖 LINKWARDEN ENHANCER - Main Menu")
        print("="*60)
        
        options = [
            ("1", "🔄 Process bookmarks (safe cleanup)", "process"),
            ("2", "📥 Import from sources", "import"),
            ("3", "🧠 View learning statistics", "stats"),
            ("4", "⚙️ Configuration", "config"),
            ("5", "📊 Generate reports", "reports"),
            ("6", "🔍 Validate data", "validate"),
            ("7", "💾 Backup & Recovery", "backup"),
            ("8", "❓ Help", "help"),
            ("q", "🚪 Quit", "quit")
        ]
        
        for key, description, _ in options:
            print(f"{key}. {description}")
        
        while True:
            try:
                choice = input("\nSelect option: ").strip().lower()
                
                for key, _, action in options:
                    if choice == key.lower():
                        return action
                
                print("❌ Invalid choice. Please try again.")
                
            except KeyboardInterrupt:
                return "quit"
    
    def show_submenu(self, title: str, options: List[Tuple[str, str, str]]) -> str:
        """Show a submenu with given options"""
        
        print(f"\n{title}")
        print("-" * len(title))
        
        for key, description, _ in options:
            print(f"{key}. {description}")
        
        print("b. ⬅️ Back to main menu")
        
        while True:
            try:
                choice = input("\nSelect option: ").strip().lower()
                
                if choice == 'b':
                    return "back"
                
                for key, _, action in options:
                    if choice == key.lower():
                        return action
                
                print("❌ Invalid choice. Please try again.")
                
            except KeyboardInterrupt:
                return "back"
    
    def get_file_path(self, prompt: str, must_exist: bool = True) -> Optional[str]:
        """Get file path from user with validation"""
        
        while True:
            try:
                path = input(f"{prompt}: ").strip()
                
                if not path:
                    return None
                
                from pathlib import Path
                path_obj = Path(path)
                
                if must_exist and not path_obj.exists():
                    print(f"❌ File does not exist: {path}")
                    continue
                
                return path
                
            except KeyboardInterrupt:
                return None
    
    def get_yes_no(self, prompt: str, default: bool = True) -> bool:
        """Get yes/no input from user"""
        
        default_text = "Y/n" if default else "y/N"
        
        while True:
            try:
                response = input(f"{prompt} ({default_text}): ").strip().lower()
                
                if not response:
                    return default
                
                if response in ['y', 'yes']:
                    return True
                elif response in ['n', 'no']:
                    return False
                else:
                    print("Please enter 'y' for yes or 'n' for no")
            
            except KeyboardInterrupt:
                return False
    
    def show_list_selection(self, title: str, items: List[str], 
                           multi_select: bool = False) -> List[int]:
        """Show list for user selection"""
        
        print(f"\n{title}")
        print("-" * len(title))
        
        for i, item in enumerate(items, 1):
            print(f"{i}. {item}")
        
        if multi_select:
            print("\nEnter numbers separated by commas (e.g., 1,3,5) or 'all' for all items:")
        else:
            print(f"\nSelect item (1-{len(items)}):")
        
        while True:
            try:
                choice = input("Selection: ").strip()
                
                if not choice:
                    return []
                
                if multi_select and choice.lower() == 'all':
                    return list(range(len(items)))
                
                if multi_select:
                    # Parse comma-separated numbers
                    try:
                        numbers = [int(n.strip()) - 1 for n in choice.split(',')]
                        valid_numbers = [n for n in numbers if 0 <= n < len(items)]
                        
                        if valid_numbers:
                            return valid_numbers
                        else:
                            print("❌ No valid selections")
                    except ValueError:
                        print("❌ Invalid format. Use numbers separated by commas")
                else:
                    # Single selection
                    try:
                        number = int(choice) - 1
                        if 0 <= number < len(items):
                            return [number]
                        else:
                            print(f"❌ Please enter a number between 1 and {len(items)}")
                    except ValueError:
                        print("❌ Please enter a valid number")
            
            except KeyboardInterrupt:
                return []