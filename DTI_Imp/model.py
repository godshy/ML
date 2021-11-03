import os, time, sys
import pickle
import tensorflow
from absl import app
from absl import flags
from absl import logging
import signal



FLAGS = flags.FLAGS

flags.DEFINE_string('echo', None, 'Text to echo.')


def main(argv):
    del argv  # Unused.

    print('Running under Python {0[0]}.{0[1]}.{0[2]}'.format(sys.version_info),
          file=sys.stderr)
    logging.info('echo is %s.', FLAGS.echo)


if __name__ == '__main__':
    app.run(main)