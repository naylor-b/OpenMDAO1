
from distutils.core import setup
import sys
import traceback
import logging

try:
    from Cython.Build import cythonize
    try:
        ext_modules = cythonize("openmdao/util/speedups/*.pyx")
    except:
        # Note, the following message won't show up during pip install unless -v arg is used
        logging.error("%s\nERROR: cython compilation of speedups failed." % traceback.format_exc())
        ext_modules = []
except ImportError:
    ext_modules = []

setup(name='openmdao',
      version='1.7.1',
      description="OpenMDAO v1 framework infrastructure",
      long_description="""\
      """,
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: Implementation :: CPython',
      ],
      keywords='optimization multidisciplinary multi-disciplinary analysis',
      author='OpenMDAO Team',
      author_email='openmdao@openmdao.org',
      url='http://openmdao.org',
      download_url='http://github.com/OpenMDAO/OpenMDAO/tarball/1.7.1',
      license='Apache License, Version 2.0',
      packages=[
          'openmdao',
          'openmdao.core',
          'openmdao.core.test',
          'openmdao.util',
          'openmdao.util.test',
          'openmdao.test',
          'openmdao.test.test',
          'openmdao.units',
          'openmdao.units.test',
          'openmdao.components',
          'openmdao.components.test',
          'openmdao.drivers',
          'openmdao.drivers.test',
          'openmdao.examples',
          'openmdao.examples.test',
          'openmdao.solvers',
          'openmdao.solvers.test',
          'openmdao.recorders',
          'openmdao.recorders.test',
          'openmdao.devtools',
          'openmdao.surrogate_models',
          'openmdao.surrogate_models.nn_interpolators',
          'openmdao.surrogate_models.test'
      ],
      package_data={
          'openmdao.units': ['unit_library.ini'],
          'openmdao.devtools': ['*.template', '*.html'],
          'openmdao.util': ['*.html'],
      },
      ext_modules=ext_modules,
      install_requires=[
        'six', 'numpydoc', 'networkx==1.11', 'numpy>=1.9.2',
        'scipy', 'sqlitedict', 'pyparsing'
      ],
      entry_points="""
      [console_scripts]
      wingproj=openmdao.devtools.wingproj:run_wing
      webview=openmdao.devtools.webview:webview_argv
      view_profile=openmdao.util.profile:prof_view
      proftotals=openmdao.util.profile:prof_totals
      profdump=openmdao.util.profile:prof_dump
      """
)
