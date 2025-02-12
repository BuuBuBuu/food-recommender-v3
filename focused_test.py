import requests
import json
from datetime import datetime
import logging
import time
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'recommender_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

class FocusedTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_results = []
        self.session = requests.Session()

    def _make_request(self, query: str) -> Dict:
        """Make request with detailed error logging."""
        try:
            start_time = time.time()
            logger.info(f"Testing query: {query}")

            response = self.session.get(
                f"{self.base_url}/recommend",
                params={"query": query},
                timeout=30
            )
            duration = time.time() - start_time

            response.raise_for_status()
            result = response.json()

            # Log the response structure
            logger.info(f"Response structure for '{query}':")
            logger.info(json.dumps(result, indent=2))

            return {
                'query': query,
                'status_code': response.status_code,
                'duration': duration,
                'data': result,
                'success': True
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for query '{query}': {str(e)}")
            return {
                'query': query,
                'status_code': getattr(e.response, 'status_code', None),
                'duration': time.time() - start_time,
                'error': str(e),
                'success': False
            }

    def analyze_response(self, response: Dict) -> Dict:
        """Analyze the response structure and content."""
        analysis = {
            'query': response['query'],
            'success': response['success'],
            'response_time': f"{response['duration']:.2f}s",
            'issues': []
        }

        if not response['success']:
            analysis['issues'].append(f"Request failed: {response.get('error', 'Unknown error')}")
            return analysis

        data = response['data']

        # Check response structure
        if 'success' not in data:
            analysis['issues'].append("Missing 'success' field in response")

        # For non-food queries or error messages
        if data.get('message'):
            analysis['response_type'] = 'message'
            analysis['message'] = data['message']
            return analysis

        # For restaurant results
        if 'restaurants' in data:
            analysis['response_type'] = 'restaurants'
            restaurants = data['restaurants']
            analysis['restaurant_count'] = len(restaurants)

            if restaurants:
                # Sample the first restaurant for structure checking
                sample = restaurants[0]
                required_fields = ['name', 'address', 'predicted_score']
                missing_fields = [field for field in required_fields if field not in sample]
                if missing_fields:
                    analysis['issues'].append(f"Missing required fields in restaurant data: {missing_fields}")

        return analysis

    def run_focused_tests(self):
        """Run focused test cases covering key scenarios."""
        test_cases = [
            # Test generic food queries
            {
                'query': "Food in Singapore",
                'description': "Generic food query",
                'expected_type': 'message'
            },
            # Test multi-country queries
            {
                'query': "Best restaurants in Singapore and Malaysia",
                'description': "Multi-country query",
                'expected_type': 'message'
            },
            # Test valid specific food queries
            {
                'query': "Chicken rice in Singapore",
                'description': "Specific food query",
                'expected_type': 'restaurants'
            },
            # Test non-food queries
            {
                'query': "What's the weather like?",
                'description': "Non-food query",
                'expected_type': 'message'
            }
        ]

        results = []
        for test_case in test_cases:
            logger.info(f"\nExecuting test case: {test_case['description']}")
            logger.info(f"Query: {test_case['query']}")

            # Make the request
            response = self._make_request(test_case['query'])

            # Analyze the response
            analysis = self.analyze_response(response)

            # Compare with expected type
            if analysis.get('response_type') != test_case['expected_type']:
                analysis['issues'].append(
                    f"Expected response type '{test_case['expected_type']}' "
                    f"but got '{analysis.get('response_type')}'"
                )

            results.append(analysis)
            self._print_test_results(analysis)

            # Wait between requests to respect API limits
            time.sleep(2)

        return results

    def _print_test_results(self, results: Dict):
        """Print formatted test results."""
        print("\n" + "="* 80)
        print(f"Test Case: {results['query']}")
        print(f"Response Time: {results['response_time']}")
        print("-" * 80)

        if results.get('message'):
            print(f"\nMessage Response: {results['message']}")
        elif results.get('restaurant_count'):
            print(f"\nRestaurants Found: {results['restaurant_count']}")

        if results['issues']:
            print("\nIssues Found:")
            for issue in results['issues']:
                print(f"- {issue}")
        else:
            print("\nNo issues found")

if __name__ == "__main__":
    print("Starting focused validation tests...")

    # Create tester instance
    tester = FocusedTester()

    # Run tests
    results = tester.run_focused_tests()

    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'test_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nTest complete. Results saved to test_results_{timestamp}.json")
