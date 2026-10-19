//> using scala 2.12.18
//> using jvm 17
//> using dep org.apache.spark::spark-sql:3.5.3
//> using dep org.jsoup:jsoup:1.18.3
//> using dep org.json:json:20240303
//> using test.dep org.scalameta::munit:1.0.2

// Spark 3.5 + JDK 17 本地运行所需参数；spark-submit 在集群会自行设置
//> using javaOpt --add-opens=java.base/java.lang=ALL-UNNAMED
//> using javaOpt --add-opens=java.base/java.lang.invoke=ALL-UNNAMED
//> using javaOpt --add-opens=java.base/java.lang.reflect=ALL-UNNAMED
//> using javaOpt --add-opens=java.base/java.io=ALL-UNNAMED
//> using javaOpt --add-opens=java.base/java.net=ALL-UNNAMED
//> using javaOpt --add-opens=java.base/java.nio=ALL-UNNAMED
//> using javaOpt --add-opens=java.base/java.util=ALL-UNNAMED
//> using javaOpt --add-opens=java.base/java.util.concurrent=ALL-UNNAMED
//> using javaOpt --add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED
//> using javaOpt --add-opens=java.base/sun.nio.ch=ALL-UNNAMED
//> using javaOpt --add-opens=java.base/sun.nio.cs=ALL-UNNAMED
//> using javaOpt --add-opens=java.base/sun.security.action=ALL-UNNAMED
//> using javaOpt --add-opens=java.base/sun.util.calendar=ALL-UNNAMED
//> using javaOpt -Djdk.reflect.useDirectMethodHandle=false
