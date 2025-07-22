def pytest_addoption(parser):
    parser.addoption("--norma", action="store", default="norma", help="Description of my argument")
