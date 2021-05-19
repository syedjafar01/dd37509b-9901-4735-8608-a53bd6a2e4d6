import razor.flow as rf

import logging
@rf.block(executor=rf.ContainerExecutor(cores=1, memory=1024))
class HelloJarvis:
    __publish__ = True
    __label__ = "hello_jarvis"
    __category__ = "Jarvis"


    message: str = "Deploy Mark42"
    out_message: rf.Output[str]

    def run(self):
        print("run method")
        print(self.message)
        self.out_message.put(self.message)