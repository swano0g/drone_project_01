#test/test_drone.py
#to run all tests in test folder simply run the command:
# python -m unittest discover -s test -p 'test_*.py'

#the following script can be used to run the specified tests modules
import unittest

if __name__ == '__main__':
    # Create a test suite combining test cases from all test unit files
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Load tests from test_drone_connection.py
    suite.addTests(loader.loadTestsFromName('test_drone_connection'))
    suite.addTests(loader.loadTestsFromName('test_drone_camera'))
    suite.addTests(loader.loadTestsFromName('test_drone_gimbal'))
    suite.addTests(loader.loadTestsFromName('test_drone_topic'))
    suite.addTests(loader.loadTestsFromName('test_drone_move'))
    suite.addTests(loader.loadTestsFromName('test_drone_home'))
    suite.addTests(loader.loadTestsFromName('test_drone_followme'))
    suite.addTests(loader.loadTestsFromName('test_drone_POI'))

    # Run the test suite
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate status code
    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)
