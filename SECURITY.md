# Security Policy

## Supported Versions

I actively support the following versions of Linkwarden Enhancer with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

I take security vulnerabilities seriously. If you discover a security vulnerability in Linkwarden Enhancer, please report it responsibly.

### How to Report

1. **Do NOT create a public GitHub issue** for security vulnerabilities
2. **Email me directly** at: [coff33ninja69@gmail.com] (or create a private issue)
3. **Include the following information**:
   - Description of the vulnerability
   - Steps to reproduce the issue
   - Potential impact assessment
   - Suggested fix (if you have one)

### What to Expect

- **Acknowledgment**: I will acknowledge receipt of your report within 48 hours
- **Initial Assessment**: I will provide an initial assessment within 5 business days
- **Regular Updates**: I will keep you informed of our progress
- **Resolution**: I aim to resolve critical vulnerabilities within 30 days

### Responsible Disclosure

I follow responsible disclosure practices:
- I will work with you to understand and resolve the issue
- I will credit you for the discovery (unless you prefer to remain anonymous)
- I will coordinate the disclosure timeline with you
- I will not take legal action against researchers who follow this policy

## Security Measures

### Data Protection

Linkwarden Enhancer implements several security measures to protect your data:

1. **Local Processing**: All bookmark processing happens locally on your machine
2. **No Data Transmission**: Personal bookmark data is never transmitted to external servers
3. **Secure File Handling**: All file operations use secure practices with proper validation
4. **Credential Protection**: API keys and tokens are stored securely and never logged

### Input Validation

- All user inputs are validated and sanitized
- File uploads are restricted by type and size
- URL validation prevents malicious redirects
- JSON parsing includes safety checks

### File System Security

- Temporary files are created with restricted permissions
- File paths are validated to prevent directory traversal
- Backup files are created with appropriate permissions
- Cache files are automatically cleaned up

### Network Security

- All HTTP requests include proper timeout handling
- SSL/TLS verification is enforced for external requests
- Rate limiting prevents abuse of external APIs
- User-Agent strings are properly set

### Dependency Security

- Dependencies are regularly updated
- Security advisories are monitored
- Vulnerable dependencies are promptly replaced
- Minimal dependency principle is followed

## Security Best Practices for Users

### Installation Security

1. **Download from Official Sources**: Only download from the official GitHub repository
2. **Verify Checksums**: Verify file integrity when possible
3. **Use Virtual Environments**: Always use Python virtual environments
4. **Keep Updated**: Regularly update to the latest version

### Configuration Security

1. **Protect API Keys**: Never commit API keys to version control
2. **Use Environment Variables**: Store sensitive configuration in environment variables
3. **Restrict File Permissions**: Ensure configuration files have appropriate permissions
4. **Regular Rotation**: Rotate API keys and tokens regularly

### Operational Security

1. **Backup Encryption**: Consider encrypting backup files
2. **Network Security**: Use secure networks when processing sensitive data
3. **Access Control**: Limit access to bookmark data and configuration files
4. **Monitoring**: Monitor for unusual activity or errors

### Data Privacy

1. **Personal Data**: Be aware of what personal data is in your bookmarks
2. **Sharing**: Be cautious when sharing processed bookmark files
3. **Cleanup**: Regularly clean up temporary and cache files
4. **Anonymization**: Consider anonymizing data before sharing for support

## Known Security Considerations

### Web Scraping

- Web scraping may expose your IP address to target websites
- Some websites may block or rate-limit scraping attempts
- Scraped content is cached locally and should be protected

### GitHub Integration

- GitHub tokens provide access to your repositories
- Tokens should be created with minimal required permissions
- Revoke tokens when no longer needed

### AI/ML Features

- Local LLM models may process sensitive bookmark content
- Ensure Ollama and other AI tools are properly configured
- Be aware of what data is processed by AI components

### File Operations

- Large file processing may consume significant system resources
- Temporary files are created during processing
- Backup files contain copies of your bookmark data

## Incident Response

In the event of a security incident:

1. **Immediate Response**: I will assess and contain the issue
2. **User Notification**: Affected users will be notified promptly
3. **Mitigation**: I will provide mitigation steps and patches
4. **Post-Incident**: I will conduct a post-incident review and improve our processes

## Security Updates

Security updates will be:
- Released as soon as possible after discovery
- Clearly marked as security updates
- Accompanied by detailed security advisories
- Backported to supported versions when necessary

## Contact Information

For security-related questions or concerns:
- **Security Issues**: [Create a private issue or email]
- **General Security Questions**: [GitHub Discussions]
- **Documentation Issues**: [GitHub Issues]

## Acknowledgments

I thank the security research community for helping keep Linkwarden Enhancer secure. Security researchers who responsibly disnerabilities wied in our security advisoss they prefer to remain anonymous).

---

**Last Updated**: January 28,
**Next Review**: April 28