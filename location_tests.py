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
        logging.FileHandler(f'location_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

class LocationTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()

        # Updated test cases to match API requirements
        self.test_cases = [
            # Large Country Tests (USA) - City Level
            {
                'category': 'Large Country - Major City',
                'query': "Pizza in New York, USA",
                'min_results': 5,
                'description': "Major US city search"
            },
            {
                'category': 'Large Country - Different City',
                'query': "Tacos in Los Angeles, USA",
                'min_results': 5,
                'description': "US West Coast city search"
            },

            # Large Country - District Level
            {
                'category': 'Large Country - District',
                'query': "Burgers in Manhattan, New York, USA",
                'min_results': 3,
                'description': "US city district search"
            },
            {
                'category': 'Large Country - Different District',
                'query': "Sushi in Brooklyn, New York, USA",
                'min_results': 3,
                'description': "US city district search alternative"
            },

            # Small Country Tests (Singapore)
            {
                'category': 'Small Country - Specific Dish',
                'query': "Chicken Rice in Singapore",
                'min_results': 5,
                'description': "Popular local dish in small country"
            },
            {
                'category': 'Small Country - District Level',
                'query': "Laksa in Geylang, Singapore",
                'min_results': 3,
                'description': "District level in small country"
            },
            {
                'category': 'Small Country - Different Area',
                'query': "Satay in Tanjong Pagar, Singapore",
                'min_results': 3,
                'description': "Different area in small country"
            },

            # Dense Urban Area Tests
            {
                'category': 'Dense Urban - City Level',
                'query': "Ramen in Tokyo, Japan",
                'min_results': 5,
                'description': "Dense urban city search"
            },
            {
                'category': 'Dense Urban - District Level',
                'query': "Sushi in Shinjuku, Tokyo, Japan",
                'min_results': 3,
                'description': "Dense urban district search"
            },

            # Border Region Tests
            {
                'category': 'Border Region',
                'query': "Seafood in Woodlands, Singapore",
                'min_results': 3,
                'description': "Location near country border"
            }
        ]

    def _make_request(self, query: str) -> Dict:
        """Make request with error handling and timing."""
        start_time = time.time()
        try:
            logger.info(f"Testing query: {query}")

            response = self.session.get(
                f"{self.base_url}/recommend",
                params={"query": query},
                timeout=30
            )
            duration = time.time() - start_time

            response.raise_for_status()
            result = response.json()

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

    def analyze_results(self, response: Dict, test_case: Dict) -> Dict:
        """Analyze test results with location-specific checks."""
        analysis = {
            'category': test_case['category'],
            'query': test_case['query'],
            'description': test_case['description'],
            'success': response['success'],
            'response_time': f"{response['duration']:.2f}s",
            'issues': []
        }

        if not response['success']:
            analysis['issues'].append(f"Request failed: {response.get('error', 'Unknown error')}")
            return analysis

        data = response['data']

        # Handle case where API returns a message instead of results
        if data.get('message'):
            analysis['message'] = data['message']
            analysis['issues'].append(f"API returned message: {data['message']}")
            return analysis

        restaurants = data.get('restaurants', [])

        # Check minimum results
        actual_results = len(restaurants)
        analysis['result_count'] = actual_results
        if actual_results < test_case['min_results']:
            analysis['issues'].append(
                f"Expected at least {test_case['min_results']} results, got {actual_results}"
            )

        # Location analysis
        if restaurants:
            locations = []
            for restaurant in restaurants:
                loc = restaurant.get('location', {})
                if loc.get('lat') and loc.get('lng'):
                    locations.append((loc['lat'], loc['lng']))

            if locations:
                # Calculate result spread
                lats, lngs = zip(*locations)
                lat_spread = max(lats) - min(lats)
                lng_spread = max(lngs) - min(lngs)

                analysis['location_stats'] = {
                    'lat_spread': lat_spread,
                    'lng_spread': lng_spread,
                    'location_count': len(locations)
                }

                # Check for result clustering
                if len(locations) > 1:
                    if lat_spread < 0.01 and lng_spread < 0.01:
                        analysis['issues'].append("Results appear to be too tightly clustered")

        return analysis

    def run_tests(self):
        """Run all location test cases."""
        results = []

        for test_case in self.test_cases:
            logger.info(f"\nExecuting test case: {test_case['description']}")

            # Make request
            response = self._make_request(test_case['query'])

            # Analyze results
            analysis = self.analyze_results(response, test_case)
            results.append(analysis)

            # Print results
            self._print_test_results(analysis)

            # Rate limiting pause
            time.sleep(2)

        return results

    def _print_test_results(self, results: Dict):
        """Print formatted test results."""
        print("\n" + "="* 80)
        print(f"Category: {results['category']}")
        print(f"Test Case: {results['description']}")
        print(f"Query: {results['query']}")
        print(f"Response Time: {results['response_time']}")
        print("-" * 80)

        if 'message' in results:
            print(f"\nAPI Message: {results['message']}")
        elif 'result_count' in results:
            print(f"\nResults Found: {results['result_count']}")

            if 'location_stats' in results:
                stats = results['location_stats']
                print("\nLocation Coverage:")
                print(f"- Number of locations: {stats['location_count']}")
                print(f"- Latitude spread: {stats['lat_spread']:.4f}")
                print(f"- Longitude spread: {stats['lng_spread']:.4f}")

        if results['issues']:
            print("\nIssues Found:")
            for issue in results['issues']:
                print(f"- {issue}")
        else:
            print("\nNo issues found")

def run_location_tests(base_url: str = "http://localhost:8000"):
    """Main function to run location tests."""
    print("Starting location-based validation tests...")

    tester = LocationTester(base_url)
    results = tester.run_tests()

    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'location_test_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    total_tests = len(results)
    failed_tests = sum(1 for r in results if r['issues'])
    print("\nTest Summary:")
    print(f"Total tests: {total_tests}")
    print(f"Successful tests: {total_tests - failed_tests}")
    print(f"Failed tests: {failed_tests}")
    print(f"\nDetailed results saved to: location_test_results_{timestamp}.json")

if __name__ == "__main__":
    run_location_tests()
