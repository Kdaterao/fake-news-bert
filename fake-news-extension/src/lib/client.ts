export async function checkText(text: string) {
    try {
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text })
        });

        return await response.json();
    } catch (err) {
        console.error(err);
        return { error: "failed to get response" };
    }
}