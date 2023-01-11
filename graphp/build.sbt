name := "Simple Project"
assemblyJarName in assembly := "simple-project_2.12-1.0.jar"

version := "1.0"
resolvers ++= Seq(
  Resolver.sonatypeRepo("releases"),
  Resolver.sonatypeRepo("snapshots"),
  Resolver.mavenLocal
)

scalaVersion := "2.12.15"

libraryDependencies += "org.apache.spark" %% "spark-sql" % "3.2.0" % "provided"
libraryDependencies += "org.apache.spark" %% "spark-core" % "3.2.0" % "provided"
libraryDependencies += "org.apache.spark" %% "spark-graphx" % "3.2.0" % "provided"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "3.2.0" % "provided"

libraryDependencies ++= Seq(
//   "com.chuusai" %% "shapeless" % "2.3.3",
  "net.liftweb" %% "lift-json" % "3.5.0",
  "com.lihaoyi" %% "upickle" % "1.5.0",
  "org.zeromq" % "jzmq" % "3.1.1-SNAPSHOT"
  // Last stable release
//   "org.scalanlp" %% "breeze" % "2.0.1-RC1",
)