// pages/index.tsx
import { useState } from 'react';
import {
  Container, Textarea, Button, Title, Text, Loader, Box
} from '@mantine/core';

export default function Home() {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(false);

  const ask = async () => {
    setLoading(true);
    const res = await fetch('http://localhost:8000/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, hybrid: true, filter: '' }),
    });
    const data = await res.json();
    setAnswer(data.answer);
    setLoading(false);
  };

  return (
    <Container size="sm" pt={40}>
      <Title mb="md">🔍 System Safety AI Assistant</Title>
      <Textarea
        placeholder="Type your question here..."
        value={question}
        onChange={(e) => setQuestion(e.currentTarget.value)}
        minRows={3}
        autosize
        mb="md"
      />
      <Button onClick={ask} loading={loading}>Submit</Button>
      {answer && (
        <Box mt="xl" p="md" bg="gray.1" style={{ borderRadius: 8 }}>
          <Text fw={500}>💡 Answer:</Text>
          <Text mt="sm" style={{ whiteSpace: 'pre-wrap' }}>{answer}</Text>
        </Box>
      )}
    </Container>
  );
}
