import asyncio
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent

#main function :
async def main():
    #1: creating multi mcp client server that uses stdio, and http transport.
    multi_server_client = MultiServerMCPClient(
    {
        "context7":{
            "url":"https://mcp.context7.com/mcp",
            "transport":"streamable_http"
        },
        "met-museum":{
            "command":"npx",
            "args": ['-y', 'metmuseum-mcp'],
            "transport":"stdio"       
        }
    }
    )
    #2:getting tools 
    tools = await multi_server_client.get_tools()
    if tools :
        #3: LLM
        ollama_model = ChatOllama(
            model="deepseek-v3.1:671b-cloud"
        )
        #4: Memory
        checkpointer = InMemorySaver()

        #5: agent = LLM + Memory:
        agent = create_agent(
            model=ollama_model,
            tools=tools,
            checkpointer=checkpointer
        )

        #6: calling the agent :
        while True:
            choice = input("""
                    if we want to talk with the agent choose (1), if you want to quit choose (2)
                    """)
            if choice == "1":
                print("Your question")
                query = input("> ")
                response = agent.invoke(
                    {
                        "messages":query
                    },
                    config = {"configurable": {"thread_id": "conversation_id"}}
                )
                print(response['messages'][-1].content)
                
            else:
                print("GoodBye!")
                break
    else:
        print("Failed  to get tools.")

#run the program:
if __name__ == "__main__":
    asyncio.run(main())

    