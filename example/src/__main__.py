import logging

import uvicorn

LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


def main():
    logging.basicConfig(level=logging.DEBUG)
    uvicorn.run(
        'example.main:api',
        host='localhost',
        port=8000,
        log_config={
            'version': 1,
            'formatters': {
                'default': {
                    'format': LOG_FORMAT,
                },
            },
            'handlers': {
                'default': {
                    'class': 'logging.StreamHandler',
                    'formatter': 'default',
                },
            },
            'loggers': {
                'uvicorn.error': {
                    'level': 'DEBUG',
                },
                'uvicorn.access': {
                    'level': 'DEBUG',
                },
            },
            'root': {
                'level': 'DEBUG',
                'handlers': ['default'],
            },
        }
    )


if __name__ == '__main__':
    main()
