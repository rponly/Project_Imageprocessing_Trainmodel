import React, { Component } from 'react'
import { Navbar } from 'react-bootstrap'

export class NavigationBar extends Component {
    render() {
        return (
            <Navbar bg="light">
                <Navbar.Brand>Signature Verification Project</Navbar.Brand>
            </Navbar>
        )
    }
}

export default NavigationBar
