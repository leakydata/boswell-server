"""Start the terminal screen headless, open the pairing screen, save a picture of each."""
import asyncio
from boswell_server.tui import ServerApp, PairScreen

async def main():
    from boswell_server import api
    api.engine.speech()
    app = ServerApp()
    async with app.run_test(size=(110, 40)) as pilot:
        await pilot.pause(25)
        app.save_screenshot("/tmp/claude-1000/-home-scholyx-Documents-electronics-boswell-phone/e83dba6d-3f28-4c34-8bb0-cf6d25f75e8e/scratchpad/screen.svg")
        await pilot.press("p")
        await pilot.pause(1)
        print("pair screen open:", isinstance(app.screen, PairScreen))
        app.save_screenshot("/tmp/claude-1000/-home-scholyx-Documents-electronics-boswell-phone/e83dba6d-3f28-4c34-8bb0-cf6d25f75e8e/scratchpad/pair.svg")

if __name__ == "__main__":
    asyncio.run(main())
