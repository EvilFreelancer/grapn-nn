RELATIONS = {
    "php": [
        "laravel", "symfony", "composer", "dockerfile", "phpunit", "xdebug", "phpstorm", "xhprof", "phpspec",
        "php-cs-fixer", "phpmd", "phpstan", "phpdoc", "phpcs", "phpcbf", "phpmetrics", "phploc", "phpcpd",
        "codeception", "behat", "docker", "crm", "git"
    ],
    "symfony": [
        "php", "api",
        "doctrine", "mysql", "redis", "postgresql", "sqlite", "mssql", "rabbitmq",
        "twig",
    ],
    "sql": ["active directory"],
    "wordpress": [
        "php",
        "woocommerce", "woocommerce-plugin", "woocommerce-theme", "woocommerce-extension", "woocommerce-api",
        "cms"
    ],
    "laravel": [
        "php", "api",
        "eloquent", "mysql", "redis", "postgresql", "sqlite", "mssql", "rabbitmq",
        "blade", "livewire", "laravel-mix", "laravel-echo", "laravel-valet", "passport",
        "lighthouse", "laravel-plugin", "octbercms", "laravel-nova", "laravel-horizon", "laravel-scout",
        "vue.js", "react.js"
    ],
    "codeception": [
        "api", "rest api"
    ],
    "behat": [
        "php", "laravel"
    ],
    "blade": [
        "php", "laravel"
    ],
    "lighthouse": [
        "graphql", "eloquent", "laravel", "api",
    ],
    "git": [
        "github", "gitlab", "bitbucket",
    ],
    "jira": ["confluence", "bitbucket"],
    "eloquent": ["mysql", "sqlite", "mssql", "sqlserver", "oracle", "cockroachdb", "tidb", "sql"],
    "python": [
        "crm",
        "django", "flask", "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "dockerfile", "sql",
        "conda", "pip", "virtualenv", "jupyter", "jupyter-notebook", "jupyterlab", "matplotlib", "seaborn", "plotly",
        "poetry", "pipenv", "pyenv", "pyenv-virtualenv", "scipy", "pyshark", "linux", "data science", "pyspark",
        "computer science", "git", "apache hive", "ansible"
    ],
    "swagger": ["openapi"],
    "api gateway": ["openapi", "swagger"],
    "openshift": ["iaas", "saas", "pass", "kubernetes"],
    "pyspark": ["apache spark"],
    "apache spark": ["java", "python"],
    "cython": ["python", "c", "c++"],
    "flask": [
        "python", "api", "sqlalchemy", "dockerfile", "flask-restful", "flask-socketio", "flask-graphql",
        "flask-admin"
    ],
    "django": ["python", "postgresql", "django-orm", "django-rest-framework", "dockerfile"],
    "django-orm": [
        "python", "django", "orm", "postgresql", "mysql", "sqlite", "mssql", "sqlserver", "oracle", "cockroachdb",
        "tidb", "sql"
    ],
    "java": [
        "angular.js", "android", "opensearch", "elasticsearch",
        "apache hbase", "apache cassandra", "apache hive", "apache hadoop", "apache airflow",
        "spring", "hibernate", "maven", "dockerfile", "groovy", "apache camel", "restful api", "soap",
        "postgresql", "sso", "javascript", "scala", "jvm", "scalatest", "apache nifi"
    ],
    "scala": ["scalatest"],
    "google docs": ["restful api"],
    "spring": ["mssql", "springboot", "thymeleaf"],
    "ruby": ["rails", "ror"],
    "rails": ["sqlite", "activerecord"],
    "javascript": [
        "nodejs", "react.js", "vue.js", "angular.js", "typescript", "dockerfile", "npm", "yarn", "jss", "css", "html",
        "scss", "sso"
    ],
    "vue.js": ["bootstrap", "vuetify", "quasar", "nuxt.js", "typescript", "dockerfile"],
    "sccs": ["css"],
    "jss": ["css"],
    "nodejs": ["express", "mongoose", "npm", "yarn", "dockerfile", "typescript", "typeorm"],
    "react.js": ["redux", "javascript", "dockerfile"],
    "golang": [
        "java", "gin", "beego", "echo", "gorm", "grpc", "dockerfile", "git", "prometheus", "kubernetes", "docker",
    ],
    "mongodb": ["linux", "node.js", "javascript", "typescript"],
    "rest api": ["nginx"],
    "dos": ["ados"],
    "openstack": ["openapi"],
    "gorm": ["mysql", "postgresql", "sqlite", "mssql", "sqlserver", "oracle", "cockroachdb", "tidb", "sql"],
    "html": ["css", "javascript"],
    "typescript": ["angular.js", "react.js", "vue.js", "webpack", "npm", "yarn", "dockerfile", "sql", "dockerfile"],
    "css": ["scss", "sass", "less", "bootstrap"],
    "c": ["linux", "windows", "macos", "bash", "c++", "dockerfile"],
    "c++": ["linux", "windows", "macos", "qt", "boost", "dockerfile", "dos"],
    "rust": ["linux", "windows", "macos", "cargo", "wasm", "dockerfile"],
    "orm": ["gorm", "django-orm", "eloquent", "sqlalchemy", "doctrine", "typeorm"],

    # DevOps & Cloud
    "prometheus": ["kubernetes", "docker", "linux", "grafana"],
    "dockerfile": ["devops", "docker", "docker-compose"],
    "testing": ["computer science", "verilog"],
    "computer science": [
        "vhdl",
        "compilation optimization", "hardware acceleration", "hdl", "high performance", "systemc",
    ],
    "gpu": ["hardware acceleration"],
    "devops": [
        "decsecops", "cloud native", "ansible",
        "load balancer", "api",
    ],
    "plantuml": ["uml"],
    "docker": [
        "devops", "cgroups",
        "kubernetes", "docker-compose", "dockerfile", "docker-swarm", "docker-hub", "jenkins", "travis-ci",
        "github-actions", "gocd", "gitlab-ci", "php", "python", "nodejs", "java", "ruby", "go", "rust", "c", "c++",
        "linux", "windows", "github", "gitlab", "bitbucket", "aws", "azure", "gcp", "kubernetes", "helm", "istio",
    ],
    "kubernetes": [
        "devops", "helm", "istio", "kubeflow", "k3s", "kubectl", "yaml"
    ],
    "aws": ["ec2", "s3", "lambda", "cloudformation", "cloudfront", "cloudwatch", "route53", "dynamodb", "rds"],
    "azure": ["azure-functions", "azure-web-apps", "azure-storage", "azure-devops", "azure-pipelines"],
    "gcp": ["gce", "gcs", "gke", "gcf", "gcr", "gcs", "gcf", "gcr", "gcloud", "gcloud-sdk"],
    "cicd": ["jenkins", "travis-ci", "github-actions", "gitlab-ci", "circleci", "teamcity", "bamboo", "gocd"],
    "opensearch": ["elasticsearch"],

    # Data Science & ML
    "r": ["shiny", "dplyr", "ggplot2"],
    "jupyter-notebook": ["pandas", "numpy", "scikit-learn", "tensorflow", "pytorch"],
    "data science": [
        "linux", "python", "tensorflow", "torch", "xgboost", "catboost", "machine learning",
        "data engineer", "ansible", "r", "uml",
    ],
    "postman": ["rest api"],
    "tensorflow": ["linux", "python", "gpu"],
    "torch": ["linux", "python", "gpu"],
    "big data": ["data science", "apache hive", "data mining", "apache hadoop", "clickstream"],

    # Mobile development
    "android": ["kotlin", "java", "android-studio"],
    "ios": ["swift", "objective-c", "cocoa-touch", "xcode"],

    # System programming & OS
    "linux": [
        "bash", "zsh", "shell", "systemd", "debian", "ubuntu", "centos", "redhat", "archlinux", "gentoo",
        "rabbitmq", "qemu", "kms", "gstreamer", "u-boot", "cgroups", "snmp", "ansible",
    ],
    "macos": ["zsh", "shell", "ios", "swift", "objective-c", "homebrew", "macos-catalina"],
    "windows": ["dos", "active directory", "mssql server"],
    "shell": [
        "bash", "zsh", "linux", "macos", "windows", "powershell", "shell-script", "shell-commands", "npm", "yarn"
    ],

    # Databases
    "databases": ["postgresql", "mysql", "sqlite", "mssql server", "oracle", "cockroachdb", "tidb", "sql"],

    # Linux
    "deb": ["linux", "debian", "ubuntu", "linuxmint", "kali-linux", "raspbian", "raspberryos", "elementaryos"],
    "rpm": ["linux", "redhat", "centos", "almalinux", "rockylinux", "fedora"],
    "arch": ["linux", "archlinux"],
    "gentoo": ["linux"],
    "alpine": ["linux"],
    "repository": ["git", "cvs", "svn", "mercurial", "github", "gitlab", "bitbucket"],
    "thin-client": ["virtualization", "vdi", "cloud"],

    # Backend programming
    "api": [
        "rest api", "restful api", "graphql api", "grpc api", "soap api", "json api", "xml api", "openapi",
        "rest", "restful", "graphql", "grpc", "soap", "json", "xml", "openapi", "swagger", "laravel", "django", "flask",
        "spring", "nodejs", "express", "gin", "beego", "php", "java", "python", "ruby"
    ],
    "api-client": [
        "rest", "rest", "graphql", "grpc", "soap", "json", "xml", "openapi", "swagger", "laravel", "django",
        "flask"
    ],
}
