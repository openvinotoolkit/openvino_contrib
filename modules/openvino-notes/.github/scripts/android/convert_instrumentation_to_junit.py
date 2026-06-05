import pathlib
import re
import sys
import xml.etree.ElementTree as ET


def arg(index: int, default: str) -> str:
    try:
        return sys.argv[index]
    except IndexError:
        return default


raw_path = pathlib.Path(arg(1, "instrumentation-raw.txt"))
output_xml = pathlib.Path(arg(2, "instrumentation-results.xml"))
output_txt = pathlib.Path(arg(3, "instrumentation.txt"))
suite_name = arg(4, "androidTest")

if not raw_path.exists():
    raise SystemExit(f"{raw_path} was not generated")

raw = raw_path.read_text()
suite = ET.Element("testsuite", name=suite_name, tests="0", failures="0", errors="0", skipped="0")
tests = failures = 0
current_class = "unknown"
current_test = "unknown"
current_stack = ""

for line in raw.splitlines():
    line = line.strip()
    m_class = re.match(r"INSTRUMENTATION_STATUS: class=(.+)", line)
    m_test = re.match(r"INSTRUMENTATION_STATUS: test=(.+)", line)
    m_stack = re.match(r"INSTRUMENTATION_STATUS: stack=(.+)", line)
    m_code = re.match(r"INSTRUMENTATION_STATUS_CODE: (-?\d+)", line)

    if m_class:
        current_class = m_class.group(1)
    elif m_test:
        current_test = m_test.group(1)
    elif m_stack:
        current_stack = m_stack.group(1)
    elif m_code:
        code = int(m_code.group(1))
        if code in (0, -2):
            tests += 1
            testcase = ET.SubElement(
                suite,
                "testcase",
                classname=current_class,
                name=current_test,
            )
            if code == -2:
                failures += 1
                failure = ET.SubElement(testcase, "failure", message="Instrumentation failure")
                failure.text = current_stack or "Unknown instrumentation failure"
                current_stack = ""

suite.set("tests", str(tests))
suite.set("failures", str(failures))
tree = ET.ElementTree(suite)
ET.indent(tree, space="  ")
tree.write(output_xml, encoding="utf-8", xml_declaration=True)
output_txt.write_text(raw)
