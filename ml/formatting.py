import logging
from string import Template


# https://www.lihaoyi.com/post/BuildyourownCommandLinewithANSIescapecodes.html#256-colors

def color_256(number):
    """
    Get a 256-bit color
    :param number: a number between 0 and 255
    :return:
    """
    return f'\x1b[38;5;{number}m'


grey = '\x1b[38;21m'
dark_grey = color_256(243)
blue = color_256(81)
yellow = '\x1b[38;5;226m'
red = '\x1b[38;5;196m'
bold_red = '\x1b[31;1m'
reset = '\x1b[0m'
green = '\x1b[38;5;113m'


class ColorConsoleFormatter(logging.Formatter):
    """Logging colored formatter, adapted from https://stackoverflow.com/a/56944256/3638629"""

    def __init__(self, fmt):
        super().__init__()
        self.fmt = fmt
        self.FORMATS = {
            logging.DEBUG: dark_grey + self.fmt + reset,
            logging.INFO: grey + self.fmt + reset,
            logging.WARNING: yellow + self.fmt + reset,
            logging.ERROR: red + self.fmt + reset,
            logging.CRITICAL: bold_red + self.fmt + reset
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def color_text(text, color):
    return color + text + reset


def printClassifierLatex(classifiers, vectorizers):
    print("\pgfplotstableread[row sep=\\\\,col sep=&]{")
    print("Classifier & Precision & Recall & F1-score \\\\")
    for classifier in classifiers:
        for vectorizer in vectorizers:
            template = Template("$classifier & $precision & $recall & $f1 \\\\")
            template_string = template.substitute(
                classifier=vectorizer.name + " " + classifier["short_name"],
                precision=round(classifier[vectorizer.name + "precision"], 3),
                recall=round(classifier[vectorizer.name + "recall"], 3),
                f1=round(classifier[vectorizer.name + "f1"], 3)
            )
            print(template_string)
    print("}\classifierdata")


def printIterationLatex(results, vectorizer, classifier):
    print("\pgfplotstableread[row sep=\\\\,col sep=&]{")
    print("Size & Precision & Recall & F1-score \\\\")

    for row in results.iterrows():
        template = Template("$size & $precision & $recall & $f1 \\\\")
        template_string = template.substitute(
            size=row[1]["training size"],
            precision=row[1]["avg_precision"],
            recall=row[1][2],  # avg_recall not working?
            f1=row[1]["avg_f1"]
        )
        print(template_string)
    print("}\iterationdata" + vectorizer.name + classifier["short_name"])
