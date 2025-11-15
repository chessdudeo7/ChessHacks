import modal

app = modal.App()

@app.function()
def hello():
    return "Modal is working!"

@app.local_entrypoint()
def main():
    print(hello.remote())

if __name__ == "__main__":
    main()