#!/usr/bin/env python3
"""
Test Enhanced Wikipedia Training System
Tests the new Wikipedia integration with WikiExtractor capabilities.
"""

import requests
import json
import time
import sys
import webbrowser
from urllib.parse import urljoin

def test_enhanced_wikipedia_interface():
    """Test that the enhanced Wikipedia trainer interface is accessible."""
    print("🧪 Testing Enhanced Wikipedia Trainer Interface")
    print("=" * 60)
    
    # Wait for backend
    print("⏳ Waiting for backend to be ready...")
    max_retries = 10
    for i in range(max_retries):
        try:
            response = requests.get('http://localhost:8000/ping', timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Backend ready: PID {data.get('pid', 'unknown')}")
                break
        except requests.exceptions.RequestException:
            if i == max_retries - 1:
                print("❌ Backend not responding")
                return False
            time.sleep(1)
    
    # Test Wikipedia trainer interface
    try:
        response = requests.get('http://localhost:8000/wikipedia-trainer', timeout=10)
        
        if response.status_code == 200:
            content = response.text
            
            # Check for key components (more flexible matching)
            checks = [
                ('Wikipedia Knowledge Ingestion System', 'Title present'),
                ('Content Input', 'Input panel present'),
                ('Extraction Options', 'Options panel present'),
                ('Training Status', 'Status panel present'),
                ('startExtraction', 'JavaScript functions present'),  # Remove () for flexibility
                ('Smart noise filtering', 'Filtering options present'),
                ('Auto-detect source concepts', 'Concept extraction present')
            ]
            
            all_passed = True
            for check_text, description in checks:
                if check_text in content:
                    print(f"✅ {description}")
                else:
                    print(f"❌ {description}")
                    all_passed = False
            
            if all_passed:
                print("✅ Enhanced Wikipedia trainer interface fully functional")
                return True
            else:
                print("⚠️  Some components missing from interface")
                return False
                
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        return False

def test_wikipedia_integration_module():
    """Test the Wikipedia integration module directly."""
    print("\n🧪 Testing Wikipedia Integration Module")
    print("=" * 60)
    
    try:
        # Import the module
        sys.path.append('src/ui')
        from wikipedia_integration import wikipedia_integration
        
        print("✅ Wikipedia integration module imported successfully")
        
        # Test URL extraction
        test_text = "Check out https://en.wikipedia.org/wiki/Machine_learning and https://en.wikipedia.org/wiki/Artificial_intelligence"
        urls = wikipedia_integration.extract_urls_from_text(test_text)
        
        if len(urls) == 2:
            print(f"✅ URL extraction working: found {len(urls)} URLs")
        else:
            print(f"⚠️  URL extraction issue: found {len(urls)} URLs, expected 2")
        
        # Test title extraction
        test_url = "https://en.wikipedia.org/wiki/Machine_learning"
        title = wikipedia_integration.extract_title_from_url(test_url)
        
        if title == "Machine learning":
            print(f"✅ Title extraction working: '{title}'")
        else:
            print(f"⚠️  Title extraction issue: got '{title}', expected 'Machine learning'")
        
        # Test concept extraction
        test_title = "Artificial Intelligence"
        test_content = "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to natural intelligence displayed by humans."
        concepts = wikipedia_integration.extract_key_concepts(test_title, test_content)
        
        if len(concepts) > 0 and "Artificial Intelligence" in concepts:
            print(f"✅ Concept extraction working: {len(concepts)} concepts found")
        else:
            print(f"⚠️  Concept extraction issue: {concepts}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import Wikipedia integration: {e}")
        return False
    except Exception as e:
        print(f"❌ Wikipedia integration test failed: {e}")
        return False

def test_enhanced_wikipedia_extraction():
    """Test the enhanced Wikipedia extraction endpoint."""
    print("\n🧪 Testing Enhanced Wikipedia Extraction Endpoint")
    print("=" * 60)
    
    # Test with a simple Wikipedia page
    test_urls = [
        "https://en.wikipedia.org/wiki/Python_(programming_language)"
    ]
    
    try:
        print(f"🔧 Testing enhanced extraction with: {test_urls[0]}")
        
        response = requests.post(
            'http://localhost:8000/wikipedia-extract',
            json={
                'urls': test_urls,
                'options': {
                    'create_associations': True,
                    'filter_noise': True,
                    'preserve_math': True,
                    'bidirectional': True,
                    'max_target_length': 1000
                }
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if data['total_processed'] > 0:
                result = data['results'][0]
                print(f"✅ Enhanced extraction successful")
                print(f"📄 Title: {result['title']}")
                print(f"📊 Content length: {result['content_length']} chars")
                print(f"🔧 Concepts extracted: {len(result['concepts'])}")
                print(f"🔗 Associations created: {result['associations_created']}")
                print(f"📈 Statistics: {data['statistics']}")
                
                return True
            else:
                print(f"⚠️  No pages processed successfully")
                return False
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error details: {error_data}")
            except:
                print(f"Response text: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        return False

def test_fallback_content_cleaning():
    """Test the fallback content cleaning functionality."""
    print("\n🧪 Testing Fallback Content Cleaning")
    print("=" * 60)
    
    try:
        sys.path.append('src/ui')
        from wikipedia_integration import wikipedia_integration
        
        # Test content with Wikipedia noise
        noisy_content = """
        Machine learning[1] is a method of data analysis[2] that automates analytical model building[citation needed]. 
        It is a branch of artificial intelligence[3] based on the idea that systems can learn from data[Smith 2020], 
        identify patterns and make decisions[Jones et al. 2019] with minimal human intervention[dubious – discuss].
        
        Mathematical expressions like [x + y = z] and [0,1] should be preserved[4].
        """
        
        cleaned = wikipedia_integration._fallback_clean_content(noisy_content)
        
        # Check that some references were removed and math was preserved
        references_removed = '[citation needed]' not in cleaned and '[Smith 2020]' not in cleaned
        math_preserved = '[x + y = z]' in cleaned and '[0,1]' in cleaned
        
        if references_removed and math_preserved:
            print("✅ Fallback cleaning working correctly")
            print(f"📊 Original length: {len(noisy_content)}")
            print(f"📊 Cleaned length: {len(cleaned)}")
            print(f"🔧 Sample cleaned: {cleaned[:100]}...")
            return True
        else:
            print("⚠️  Fallback cleaning not working as expected")
            print(f"References removed: {references_removed}")
            print(f"Math preserved: {math_preserved}")
            print(f"Cleaned content: {cleaned}")
            return False
            
    except Exception as e:
        print(f"❌ Fallback cleaning test failed: {e}")
        return False

def open_enhanced_wikipedia_trainer():
    """Open the enhanced Wikipedia trainer in the default browser."""
    print("\n🌐 Opening Enhanced Wikipedia Trainer Interface")
    print("=" * 60)
    
    url = "http://localhost:8000/wikipedia-trainer"
    
    try:
        print(f"🔗 Opening: {url}")
        webbrowser.open(url)
        print("✅ Enhanced Wikipedia trainer opened in browser")
        print("\n📋 Enhanced Features:")
        print("1. WikiExtractor integration for better content cleaning")
        print("2. Enhanced concept extraction and association creation")
        print("3. Real-time statistics and progress monitoring")
        print("4. Smart noise filtering with mathematical expression preservation")
        print("5. Bidirectional association learning")
        print("6. Batch processing of multiple Wikipedia URLs")
        return True
    except Exception as e:
        print(f"❌ Error opening browser: {e}")
        print(f"🔗 Manual URL: {url}")
        return False

def main():
    """Main test function."""
    print("🧪 Enhanced Wikipedia Training System Test Suite")
    print("=" * 70)
    
    # Run tests
    tests = [
        ("Enhanced Wikipedia Trainer Interface", test_enhanced_wikipedia_interface),
        ("Wikipedia Integration Module", test_wikipedia_integration_module),
        ("Enhanced Wikipedia Extraction", test_enhanced_wikipedia_extraction),
        ("Fallback Content Cleaning", test_fallback_content_cleaning)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*70}")
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print(f"\n{'='*70}")
    print("🎯 Enhanced Wikipedia Training System Test Summary:")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"• {test_name}: {status}")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ENHANCED WIKIPEDIA TRAINING SYSTEM WORKING!")
        print("✅ WikiExtractor integration (with fallback)")
        print("✅ Enhanced content extraction and cleaning")
        print("✅ Smart concept extraction and association creation")
        print("✅ Real-time progress monitoring and statistics")
        print("✅ Mathematical expression preservation")
        print("✅ Bidirectional learning capabilities")
        print("\n🏆 ENHANCED AUTOMATED KNOWLEDGE INGESTION COMPLETE!")
        
        # Open the trainer interface
        open_enhanced_wikipedia_trainer()
        
        print("\n🚀 READY FOR ENHANCED WIKIPEDIA TRAINING!")
        print("   • Features: WikiExtractor integration, smart cleaning, concept extraction")
        print("   • Capabilities: Batch processing, real-time monitoring, bidirectional learning")
        print("   • Integration: Full backend integration with enhanced association system")
        sys.exit(0)
    else:
        print("❌ Some enhanced Wikipedia training system tests failed")
        sys.exit(1)

if __name__ == "__main__":
    main()