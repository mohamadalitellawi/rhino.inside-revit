"""Run Grasshopper tests for Rhino.Inside.Revit

Usage:
    {} [-d] [<test_name>] [<revit_year>]

Options:
    -h, --help          Show this help
    -d                  Print debug messages
    <test_name>         GH test name (e.g. "test" in "test.ghx")
    <revit_year>        Revit version to be tested (e.g. 2019)
"""
#pylint: disable=broad-except,global-statement,invalid-name
import os
import os.path as op
import re
from collections import namedtuple
import logging
from typing import List

# third-party dependencies
from docopt import docopt

# rir dependencies
import dbgrevit

# cli info
__binname__ = op.splitext(op.basename(__file__))[0] # grab script name
__version__ = '1.0'

# =============================================================================
# Configs
# =============================================================================
# test directories to be included
TESTS = [
    'ghtests'
]
# naming format gh test files
GHTEST_FORMAT = r'(.+).ghx'
# naming format for Revit models associated with tests
GHTEST_MODEL_FORMAT = r'{test}_R(\d{{2}}).rvt'
# naming format for test results file
GHTEST_RESULTS_FORMAT = '{test}_Results.txt'
RESULT_DIVIDER = ';'
RESULT_OK = 'OK'
# range of revit version to test (lowest, highest)
# test machine is expected to have these versions installed
REVIT_RANGE = (2019, 2021)
# =============================================================================

"""GH test info"""
GHTest = namedtuple('GHTest', ['name', 'dir', 'path', 'models', 'resultfile'])

"""GH test Revit model"""
GHTestModel = namedtuple('GHTestModel', ['path', 'version'])


class CLIArgs:
    """Data type to hold command line args"""
    def __init__(self, args):
        self.test_name = args['<test_name>']
        self.revit_year = args['<revit_year>']
        self.debug = args['-d']


class GHTestCase:
    """GH test case"""
    def __init__(self, ghtest: GHTest):
        self.name = ghtest.name
        self.gh_file = ghtest.path
        self.gh_resultsfile = ghtest.resultfile
        self.gh_models: List[GHTestModel] = ghtest.models
        self.has_failures = False

    def get_model(self, revit_version: int):
        """Determine appropriate test model for specified Revit version"""
        for ghmodel in self.gh_models:
            if ghmodel.version == revit_version:
                return ghmodel
        return max(self.gh_models, key=lambda x: x.version)

    def run_model_test(self, revit_version: int, ghmodel: GHTestModel):
        """Run GH test on specified Revit version and model"""
        def result_ready():
            return op.isfile(self.gh_resultsfile) \
                and os.access(self.gh_resultsfile, os.R_OK)

        logging.debug("Run test on %s", ghmodel.path)
        args = dbgrevit.CLIArgs({
            '<revit_year>': revit_version,
            '<model_path>': ghmodel.path,
            '<ghdoc_path>': self.gh_file,
            '--rps': False,
            '--dryrun': False})
        logging.debug("Waiting for test results @ %s", self.gh_resultsfile)
        dbgrevit.run_command(args, wait_until=result_ready)

    def process_results(self, revit_version: int) -> bool:
        """Process and print GH test results"""
        failed_any = False
        if op.isfile(self.gh_resultsfile):
            with open(self.gh_resultsfile, 'r') as res_file:
                results = res_file.readlines()
            for result in results:
                name, comp, res = result.strip().split(RESULT_DIVIDER)
                check = '\033[32mPASS\033[0m' if res == RESULT_OK \
                    else '\033[31mFAIL\033[0m'
                failed_any |= not res == RESULT_OK
                print(f"[ {check} ] "
                      f"version={revit_version} test={name} component={comp}")
        return failed_any

    def setUp(self):
        """Setup env before running GH test"""
        if op.isfile(self.gh_resultsfile):
            os.remove(self.gh_resultsfile)

    def run(self):
        """Run GH test case"""
        logging.debug("Starting test: \"%s\"", self.name)
        for revit_version in range(REVIT_RANGE[0], REVIT_RANGE[1] + 1):
            self.setUp()
            ghmodel = self.get_model(revit_version)
            self.run_model_test(revit_version, ghmodel)
            self.has_failures |= self.process_results(revit_version)
            self.tearDown()

    def tearDown(self):
        """Cleanup after running GH test"""
        if op.isfile(self.gh_resultsfile):
            os.remove(self.gh_resultsfile)


def find_ghtest_models(test_dir: str, test_name: str) -> List[GHTestModel]:
    """Find Revit models associated with a test"""
    logging.debug("Finding models for test \"%s\" @ %s", test_name, test_dir)
    models = []
    for entry in os.listdir(test_dir):
        m_res = re.match(GHTEST_MODEL_FORMAT.format(test=test_name), entry)
        if m_res:
            model_path = op.join(test_dir, entry)
            logging.debug("Model found @ %s", model_path)
            models.append(
                GHTestModel(
                    path=model_path,
                    version=int(m_res.groups()[0])
                )
            )
    return models


def find_ghtests(root: str) -> List[GHTest]:
    """Find GH test files and associated models in given directory"""
    logging.debug("Finding tests @ %s", root)
    ghtests: List[GHTest] = []
    for entry in os.listdir(root):
        if re.match(GHTEST_FORMAT, entry):
            ghtest_name = op.splitext(entry)[0]
            logging.debug("Test found: \"%s\"", ghtest_name)
            ghtests.append(
                GHTest(
                    name=ghtest_name,
                    dir=root,
                    path=op.join(root, entry),
                    models=find_ghtest_models(root, ghtest_name),
                    resultfile=op.join(
                        root,
                        GHTEST_RESULTS_FORMAT.format(test=ghtest_name)
                    )
                )
            )
    return ghtests


def confirm_revits() -> bool:
    """Confirm expected Revit versions are installed"""
    for revit_ver in range(REVIT_RANGE[0], REVIT_RANGE[1] + 1):
        if not dbgrevit.find_revit_binary(revit_ver):
            raise Exception(f"Expected Revit {revit_ver} is not installed")
    return True


def run_tests(test_roots: List[str], cfg: CLIArgs) -> bool:
    """Run GH tests under given directories

    Args:
        test_roots (List[str]): root directories containing tests
        cfg (CLIArgs): command line args

    Returns:
        bool: if any of the tests has failed
    """
    failed_any = False

    # prepare logger
    logging.basicConfig(
        format="DEBUG: %(message)s",
        level=logging.DEBUG if cfg.debug else logging.WARN)
    logging.debug('Requested Test: \"%s\"', cfg.test_name)
    logging.debug('Revit Version: %s', cfg.revit_year)

    # make sure env is ready
    try:
        confirm_revits()
    except Exception as testEx:
        logging.critical(str(testEx))
        exit(1)

    # find and run tests
    for testroot in test_roots:
        for gh_test in find_ghtests(testroot):
            gh_testcase = GHTestCase(gh_test)
            try:
                gh_testcase.setUp()
                gh_testcase.run()
            except Exception as test_ex:
                logging.critical(test_ex)
            finally:
                gh_testcase.tearDown()
                if gh_testcase.has_failures:
                    failed_any = True
    return failed_any


if __name__ == '__main__':
    cwd = op.abspath(op.dirname(__file__))
    try:
        # run all the tests
        # run_tests captures and logs exceptions
        has_failures = run_tests(
            # make settings from cli args
            test_roots=[op.join(cwd, x) for x in TESTS],
            cfg=CLIArgs(
                # process args
                docopt(
                    __doc__.format(__binname__),
                    version='{} {}'.format(__binname__, __version__)
                )
            )
        )
        # return appropriate errorno
        exit(0 + has_failures)

    # gracefully handle exceptions and print results
    except Exception as run_ex:
        logging.critical(run_ex)
        raise run_ex
        exit(1)


# TODO: fix run logic to it runs by revit version
